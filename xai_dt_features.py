"""
xai_dt_features.py – Analyse XAI : Feature Importance du modèle Snake
=======================================================================
Miroir de xai_features.py pour le modèle XGBoost / GradientBoosting.

3 analyses :
  1. Permutation Importance  : brouiller chaque feature → mesurer la chute de score
  2. Importance native        : gain / weight / cover XGBoost vs feature_importances_
                                sklearn — comparaison des 3 types + radar des top-8
  3. Corrélation features/actions : quelle feature déclenche quelle action ?

Features (26 valeurs) :
  [0..7]   distances murs   (N NE E SE S SW W NW)
  [8..15]  distances food   (N NE E SE S SW W NW)
  [16]     food_delta_x     [17] food_delta_y
  [18..21] danger binaire   (N E S W)
  [22..25] direction one-hot(UP RIGHT DOWN LEFT)

Usage :
    python xai_dt_features.py                   # toutes les analyses
    python xai_dt_features.py --permutation      # permutation importance
    python xai_dt_features.py --native           # importance native XGBoost
    python xai_dt_features.py --correlation      # corrélation features/actions
    python xai_dt_features.py --model snake_xgb_model.pkl
    python xai_dt_features.py --episodes 30      # épisodes (défaut : 20)
"""

import argparse
import math
import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

import pygame

warnings.filterwarnings("ignore")

import snake as game
from arbre_de_decision import DecisionTreeAgent, N_FEATURES, N_ACTIONS

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── Pygame headless ─────────────────────────────────────────────────────────
pygame.init()
game.show    = False
game.display = None

# ── Dossier de sortie ────────────────────────────────────────────────────────
OUT_DIR = "xai_dt_features"
os.makedirs(OUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    return os.path.join(OUT_DIR, filename)


# ─────────────────────────────────────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "Mur N",    "Mur NE",   "Mur E",    "Mur SE",
    "Mur S",    "Mur SW",   "Mur W",    "Mur NW",
    "Food N",   "Food NE",  "Food E",   "Food SE",
    "Food S",   "Food SW",  "Food W",   "Food NW",
    "ΔFood X",  "ΔFood Y",
    "Dang. N",  "Dang. E",  "Dang. S",  "Dang. W",
    "Dir. UP",  "Dir. R",   "Dir. DN",  "Dir. L",
]

ACTION_NAMES  = ["UP ↑", "RIGHT →", "DOWN ↓", "LEFT ←"]
ACTION_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]

BG       = "#0D1117"
PANEL_BG = "#0D1B2A"
GRID_COL = "#1E3A5F"
TEXT_COL = "#CCDDEE"

CMAP_IMPORTANCE = LinearSegmentedColormap.from_list(
    "imp", ["#0D1B2A","#1A3A5C","#1F618D","#2E86C1","#F39C12","#E74C3C"]
)
CMAP_CORR = LinearSegmentedColormap.from_list(
    "corr", ["#C0392B","#922B21","#1A1A2E","#1A5276","#2E86C1"]
)


# ─────────────────────────────────────────────────────────────────────────────
#  Utilitaires partagés
# ─────────────────────────────────────────────────────────────────────────────
def load_agent(model_path: str = "snake_xgb_model.pkl") -> DecisionTreeAgent:
    agent = DecisionTreeAgent(use_cuda=False)
    agent.load(model_path=model_path)
    # Restaure use_cuda selon le type réel du modèle chargé
    # (load() ne restaure pas ce flag depuis le pickle)
    if XGBOOST_AVAILABLE and hasattr(agent.model, "get_score"):
        agent.use_cuda = True
    if not agent.trained:
        print("[WARN] Modèle non entraîné — résultats aléatoires.")
    return agent


def predict_class(agent: DecisionTreeAgent, X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(X).astype(np.float32)
    Xs = agent.scaler.transform(X)
    if XGBOOST_AVAILABLE and hasattr(agent.model, "get_score"):
        dmat = xgb.DMatrix(Xs)
        return agent.model.predict(dmat).astype(int)
    else:
        return agent.model.predict(Xs).astype(int)


def apply_style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(PANEL_BG)
    if title:  ax.set_title(title,  color="white",  fontsize=11, fontweight="bold", pad=9)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=8)
    ax.tick_params(colors="#8899AA", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.5)


class _Collector:
    def __init__(self, agent, shuffle_feat=-1, noise_feat=-1, noise_std=0.0):
        self.agent       = agent
        self.shuffle_feat = shuffle_feat
        self.noise_feat   = noise_feat
        self.noise_std    = noise_std
        self.steps = []   # (state, action)

    def tab_state(self, *args):
        return list(args)

    def get_action(self, net, state):
        s = list(state)
        if self.shuffle_feat >= 0:
            s[self.shuffle_feat] = float(np.random.uniform(0, 1))
        if self.noise_feat >= 0 and self.noise_std > 0:
            s[self.noise_feat] = float(np.clip(
                s[self.noise_feat] + np.random.normal(0, self.noise_std), 0, 1
            ))
        if len(s) >= 26:
            if   s[22] > 0.5: self.agent.set_direction("UP")
            elif s[23] > 0.5: self.agent.set_direction("RIGHT")
            elif s[24] > 0.5: self.agent.set_direction("DOWN")
            elif s[25] > 0.5: self.agent.set_direction("LEFT")
        action = self.agent.get_action(s)
        self.steps.append((s, action))
        return action


def run_episode(agent: DecisionTreeAgent,
                shuffle_feat: int = -1) -> tuple:
    """Lance un épisode, retourne (score, states_list, actions_list)."""
    game.show           = False
    game.stop_iteration = 2000
    agent.direction     = "RIGHT"
    col = _Collector(agent, shuffle_feat=shuffle_feat)
    score = game.game_loop(
        game.rect_width, game.rect_height, game.display,
        agent, None, 0, col
    )
    states  = [s for s, a in col.steps]
    actions = [a for s, a in col.steps]
    return score, states, actions


# ─────────────────────────────────────────────────────────────────────────────
#  Analyse 1 – Permutation Importance
# ─────────────────────────────────────────────────────────────────────────────
def compute_permutation_importance(agent: DecisionTreeAgent,
                                   n_episodes: int = 20
                                   ) -> tuple:
    """
    Pour chaque feature :
      1. Jouer n_episodes épisodes avec la feature randomisée
      2. Comparer au score baseline (toutes features intactes)
    Retourne (drop_mean [F], baseline_mean, drop_std [F]).
    """
    print(f"  [PI] Baseline ({n_episodes} épisodes)…")
    baseline = [run_episode(agent)[0] for _ in range(n_episodes)]
    baseline_mean = float(np.mean(baseline))
    print(f"  [PI] Score baseline : {baseline_mean:.2f}")

    drops     = np.zeros(N_FEATURES)
    drops_std = np.zeros(N_FEATURES)

    for fi in range(N_FEATURES):
        shuffled = [run_episode(agent, shuffle_feat=fi)[0]
                    for _ in range(n_episodes)]
        mean_sh  = float(np.mean(shuffled))
        drop     = max(baseline_mean - mean_sh, 0.0)
        drops[fi]     = drop
        drops_std[fi] = float(np.std(shuffled))
        print(f"  [PI] {fi:>2} {FEATURE_NAMES[fi]:<14} "
              f"score={mean_sh:.2f}  drop={drop:+.2f}")

    return drops, baseline_mean, drops_std


def plot_permutation_importance(drops: np.ndarray, baseline: float,
                                drops_std: np.ndarray):
    n     = N_FEATURES
    order = np.argsort(drops)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor=BG,
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(
        f"Permutation Importance — score baseline : {baseline:.2f}\n"
        "Drop = chute de score quand la feature est brouillée (remplacée par valeur aléatoire)",
        fontsize=14, fontweight="bold", color="white"
    )

    # ── Barplot horizontal ──────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    norm   = Normalize(vmin=0, vmax=max(drops.max(), 1e-8))
    colors = [CMAP_IMPORTANCE(norm(drops[order[i]])) for i in range(n)]

    bars = ax.barh(range(n), drops[order], xerr=drops_std[order],
                   color=colors, edgecolor="#1A1A2E",
                   error_kw=dict(ecolor="#AAAAAA", lw=1.2, capsize=3),
                   height=0.72)

    for i, (drop, std) in enumerate(zip(drops[order], drops_std[order])):
        ax.text(drop + std + 0.02, i, f"{drop:.2f}", va="center",
                ha="left", color=TEXT_COL, fontsize=7.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in order],
                       color=TEXT_COL, fontsize=8.5)

    # Séparateurs catégories
    seps = [
        (sum(1 for i in order if i < 8),  "murs / food"),
        (sum(1 for i in order if i < 16), "food / enrichies"),
    ]
    prev = 0
    for sep_n, label in seps:
        y = n - sep_n - 0.5
        if 0 < y < n:
            ax.axhline(y=y, color="#F39C12", linewidth=1.2,
                       linestyle="--", alpha=0.7)
            ax.text(drops.max() * 0.98, y + 0.3, f"── {label} ──",
                    color="#F39C12", fontsize=7.5, ha="right", alpha=0.8)

    apply_style(ax, title="Chute de score par feature brouillée",
                xlabel="Drop de score moyen (baseline – brouillée)")

    # ── Radar chart top-8 ──────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    ax2.set_aspect("equal")

    top8       = order[:8]
    drops_top8 = drops[top8]
    vals       = drops_top8 / (drops_top8.max() + 1e-8)
    labs       = [FEATURE_NAMES[i] for i in top8]
    N_pts      = len(top8)
    angles     = [2 * math.pi * k / N_pts for k in range(N_pts)] + [0]
    vals_r     = list(vals) + [vals[0]]

    for level in [0.25, 0.5, 0.75, 1.0]:
        ring_xs = [level * math.cos(a) for a in angles]
        ring_ys = [level * math.sin(a) for a in angles]
        ax2.plot(ring_xs, ring_ys, color=GRID_COL, linewidth=0.7, alpha=0.6)
        ax2.text(level + 0.04, 0.02, f"{int(level*100)}%",
                 color="#7A9CC0", fontsize=6, va="center", alpha=0.8)

    for a in angles[:-1]:
        ax2.plot([0, math.cos(a)], [0, math.sin(a)],
                 color=GRID_COL, linewidth=0.7, alpha=0.6)

    xs = [v * math.cos(a) for v, a in zip(vals_r, angles)]
    ys = [v * math.sin(a) for v, a in zip(vals_r, angles)]
    ax2.fill(xs, ys, color="#2E86C1", alpha=0.28)
    ax2.plot(xs, ys, color="#4FC3F7", linewidth=2.2)
    ax2.scatter(xs[:-1], ys[:-1], color="#FFD700", s=70, zorder=5)

    for rank, (i, a, lab) in enumerate(zip(range(N_pts), angles[:-1], labs)):
        raw_drop  = drops_top8[i]
        feat_idx  = top8[i]
        if   feat_idx < 8:  col_lab = ACTION_COLORS[0]
        elif feat_idx < 16: col_lab = ACTION_COLORS[2]
        elif feat_idx < 18: col_lab = "#CE93D8"
        elif feat_idx < 22: col_lab = "#E74C3C"
        else:               col_lab = "#95A5A6"
        full_lab = f"#{rank+1} {lab}\ndrop={raw_drop:.2f}"
        ax2.text(1.38 * math.cos(a), 1.38 * math.sin(a), full_lab,
                 ha="center", va="center", color=col_lab, fontsize=6.8,
                 fontweight="bold", multialignment="center")

    legend_p = [
        mpatches.Patch(color=ACTION_COLORS[0], label="Distances murs   (0–7)"),
        mpatches.Patch(color=ACTION_COLORS[2], label="Distances food   (8–15)"),
        mpatches.Patch(color="#CE93D8",         label="Δ food           (16–17)"),
        mpatches.Patch(color="#E74C3C",         label="Danger binaire  (18–21)"),
        mpatches.Patch(color="#95A5A6",         label="Direction       (22–25)"),
    ]
    ax2.text(0, -1.72,
             "Rayon = chute de score normalisée\n100% = feature la plus critique",
             ha="center", va="top", color="#99AABB", fontsize=7, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0D1B2A",
                       edgecolor=GRID_COL, alpha=0.85))

    ax2.set_xlim(-1.75, 1.75)
    ax2.set_ylim(-2.50, 1.75)
    ax2.axis("off")
    ax2.set_title("Top 8 features – Radar d'importance\n"
                  "(rayon ∝ chute de score)",
                  color="white", fontsize=10, fontweight="bold", pad=12)

    legend_ax = fig.add_axes([0.68, 0.06, 0.28, 0.22])
    legend_ax.set_facecolor(PANEL_BG)
    legend_ax.axis("off")
    legend_ax.legend(handles=legend_p, fontsize=7.5, facecolor=PANEL_BG,
                     edgecolor="#444", labelcolor="white",
                     loc="center", framealpha=0.9)
    legend_ax.set_title("Catégories de features", color="white",
                        fontsize=8, pad=4)

    plt.tight_layout()
    plt.savefig(out("xai_dt_permutation.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_dt_permutation.png')}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Analyse 2 – Importance native (gain / weight / cover)
# ─────────────────────────────────────────────────────────────────────────────
def _is_xgb_booster(agent: DecisionTreeAgent) -> bool:
    """Retourne True si le modèle chargé est un XGBoost Booster."""
    return XGBOOST_AVAILABLE and hasattr(agent.model, "get_score")


def _get_native_importance(agent: DecisionTreeAgent) -> dict:
    result = {}
    if _is_xgb_booster(agent):
        for itype in ("gain", "weight", "cover"):
            d   = agent.model.get_score(importance_type=itype)
            arr = np.zeros(N_FEATURES, dtype=np.float32)
            for key, val in d.items():
                fi = int(key[1:])
                if fi < N_FEATURES:
                    arr[fi] = val
            result[itype] = arr
    else:
        result["gain"] = np.array(agent.model.feature_importances_,
                                   dtype=np.float32)
    return result


def plot_native_importance(agent: DecisionTreeAgent):
    """
    Comparaison des 3 types d'importance :
      - Barplots côte-à-côte
      - Nuage gain vs weight
      - Pie des catégories de features
    """
    imp    = _get_native_importance(agent)
    types  = list(imp.keys())
    colors = {"gain": "#F39C12", "weight": "#4FC3F7", "cover": "#81C784"}
    labels = {"gain": "Gain (pureté)", "weight": "Weight (splits)", "cover": "Cover (exemples)"}

    fig = plt.figure(figsize=(22, 10), facecolor=BG)
    fig.suptitle(
        "Importance native XGBoost — Comparaison des 3 métriques d'importance",
        fontsize=14, fontweight="bold", color="white"
    )

    gs = gridspec.GridSpec(2, len(types) + 1, figure=fig, wspace=0.38, hspace=0.50)

    for col_idx, itype in enumerate(types):
        arr   = imp[itype]
        order = np.argsort(arr)[::-1]
        norm_c = Normalize(vmin=0, vmax=max(arr.max(), 1e-8))
        bar_colors = [CMAP_IMPORTANCE(norm_c(arr[order[i]])) for i in range(N_FEATURES)]

        # ── Barplot haut ────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_facecolor(PANEL_BG)
        ax.barh(range(N_FEATURES), arr[order], color=bar_colors,
                edgecolor="#0D1117", height=0.72)
        for i, v in enumerate(arr[order]):
            if v > 0:
                ax.text(v * 1.01, i, f"{v:.1f}" if v >= 1 else f"{v:.3f}",
                        va="center", color=TEXT_COL, fontsize=6)
        ax.set_yticks(range(N_FEATURES))
        ax.set_yticklabels([FEATURE_NAMES[i] for i in order],
                           color=TEXT_COL, fontsize=7.5)
        # Séparateur murs / food
        n_above = sum(1 for i in order if i < 8)
        yline = N_FEATURES - n_above - 0.5
        if 0 < yline < N_FEATURES:
            ax.axhline(y=yline, color="#F39C12", linewidth=1.0,
                       linestyle="--", alpha=0.6)
        apply_style(ax, title=labels.get(itype, itype),
                    xlabel=f"Score {itype}")
        n_dead = (arr == 0).sum()
        ax.text(0.97, 0.02, f"Mortes : {n_dead}/{N_FEATURES}",
                transform=ax.transAxes, va="bottom", ha="right",
                color="#E74C3C" if n_dead > 0 else "#2ECC71", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0D1B2A",
                          edgecolor=GRID_COL, alpha=0.9))

    # ── Pie des catégories (haut droite) ──────────────────────────────────
    ax_pie = fig.add_subplot(gs[0, len(types)])
    ax_pie.set_facecolor(PANEL_BG)
    first_imp = imp[types[0]]
    cat_sums = [
        first_imp[0:8].sum(),
        first_imp[8:16].sum(),
        first_imp[16:18].sum(),
        first_imp[18:22].sum(),
        first_imp[22:26].sum(),
    ]
    cat_labels = ["Murs (0–7)", "Food (8–15)", "ΔFood (16–17)",
                  "Danger (18–21)", "Direction (22–25)"]
    cat_colors = [ACTION_COLORS[0], ACTION_COLORS[2],
                  "#CE93D8", "#E74C3C", "#95A5A6"]
    wedges, texts, autotexts = ax_pie.pie(
        cat_sums, labels=cat_labels, colors=cat_colors,
        autopct="%1.1f%%", startangle=140,
        textprops={"color": TEXT_COL, "fontsize": 8},
        wedgeprops={"edgecolor": "#0D1117", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_color("white"); at.set_fontsize(8)
    ax_pie.set_title(f"Part par catégorie\n({labels.get(types[0], types[0])})",
                     color="white", fontsize=10, fontweight="bold", pad=8)

    # ── Nuage gain vs weight (bas gauche) ─────────────────────────────────
    if len(types) >= 2:
        ax_sc = fig.add_subplot(gs[1, :2])
        ax_sc.set_facecolor(PANEL_BG)
        gain_arr  = imp[types[0]]
        wt_arr    = imp[types[1]]
        feat_cats = (
            ["#4FC3F7"] * 8 +   # murs
            ["#81C784"] * 8 +   # food
            ["#CE93D8"] * 2 +   # delta
            ["#E74C3C"] * 4 +   # danger
            ["#95A5A6"] * 4     # direction
        )
        ax_sc.scatter(gain_arr, wt_arr, c=feat_cats, s=80,
                      edgecolors="#222244", linewidths=0.8, zorder=3)
        for i in range(N_FEATURES):
            ax_sc.annotate(FEATURE_NAMES[i], (gain_arr[i], wt_arr[i]),
                           textcoords="offset points", xytext=(4, 3),
                           color=TEXT_COL, fontsize=6.5, alpha=0.85)
        ax_sc.axvline(x=np.median(gain_arr), color="#F39C12",
                      linestyle="--", linewidth=1, alpha=0.6)
        ax_sc.axhline(y=np.median(wt_arr), color="#F39C12",
                      linestyle="--", linewidth=1, alpha=0.6)
        ax_sc.text(gain_arr.max() * 0.5, wt_arr.max() * 0.95,
                   "forte importance", color="#F39C12", fontsize=8, alpha=0.7)
        apply_style(ax_sc,
                    title=f"Nuage {labels.get(types[0],'gain')} vs {labels.get(types[1],'weight')}",
                    xlabel=f"Score {types[0]}", ylabel=f"Score {types[1]}")
        ax_sc.grid(axis="both", color=GRID_COL, linewidth=0.4, alpha=0.4)

    # ── Heatmap normalisée [type × feature] (bas droite) ──────────────────
    ax_h = fig.add_subplot(gs[1, 2:])
    ax_h.set_facecolor(PANEL_BG)
    heat = np.zeros((len(types), N_FEATURES))
    for ti, itype in enumerate(types):
        arr = imp[itype]
        mx  = arr.max()
        heat[ti] = arr / mx if mx > 0 else arr
    feat_order = np.argsort(heat[0])[::-1]

    im = ax_h.imshow(heat[:, feat_order], cmap=CMAP_IMPORTANCE,
                     vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax_h.set_yticks(range(len(types)))
    ax_h.set_yticklabels([labels.get(t, t) for t in types],
                          color=TEXT_COL, fontsize=9)
    ax_h.set_xticks(range(N_FEATURES))
    ax_h.set_xticklabels([FEATURE_NAMES[i] for i in feat_order],
                          rotation=40, ha="right", color=TEXT_COL, fontsize=7)
    ax_h.set_title("Importance normalisée [0–1] par type × feature",
                   color="white", fontsize=10, fontweight="bold", pad=8)
    for spine in ax_h.spines.values():
        spine.set_edgecolor(GRID_COL)
    cbar = plt.colorbar(im, ax=ax_h, fraction=0.03, pad=0.02)
    cbar.set_label("Importance normalisée", color=TEXT_COL, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    plt.savefig(out("xai_dt_native.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_dt_native.png')}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Analyse 3 – Corrélation features / actions
# ─────────────────────────────────────────────────────────────────────────────
def collect_states(agent: DecisionTreeAgent, n_episodes: int = 20):
    """Collecte (states, actions) sur n_episodes épisodes greedy."""
    game.show           = False
    game.stop_iteration = 2000

    all_states, all_actions = [], []
    for ep in range(n_episodes):
        score, states, actions = run_episode(agent)
        all_states.extend(states)
        all_actions.extend(actions)
        print(f"  [Corr] Épisode {ep+1}/{n_episodes} → score {score}")
    return (np.array(all_states, dtype=np.float32),
            np.array(all_actions, dtype=np.int32))


def compute_correlation(states: np.ndarray,
                        actions: np.ndarray) -> tuple:
    """Matrice de corrélation de Pearson [N_FEATURES, N_ACTIONS]."""
    corr = np.zeros((N_FEATURES, N_ACTIONS))
    for fi in range(N_FEATURES):
        for ai in range(N_ACTIONS):
            binary = (actions == ai).astype(float)
            r, _   = pearsonr(states[:, fi], binary)
            corr[fi, ai] = r if not np.isnan(r) else 0.0

    mean_per_action = np.zeros((N_ACTIONS, N_FEATURES))
    std_per_action  = np.zeros((N_ACTIONS, N_FEATURES))
    for ai in range(N_ACTIONS):
        mask = actions == ai
        if mask.sum() > 0:
            mean_per_action[ai] = states[mask].mean(axis=0)
            std_per_action[ai]  = states[mask].std(axis=0)

    return corr, mean_per_action, std_per_action


def plot_correlation(corr: np.ndarray,
                     mean_per_action: np.ndarray,
                     std_per_action: np.ndarray):
    fig = plt.figure(figsize=(22, 13), facecolor=BG)
    fig.suptitle(
        "Corrélation Features → Actions  —  Ce qui déclenche chaque décision",
        fontsize=15, fontweight="bold", color="white"
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.42, hspace=0.50)

    # ── Heatmap [N_FEATURES × N_ACTIONS] ─────────────────────────────────
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_heat.set_facecolor(PANEL_BG)
    vabs = np.abs(corr).max()
    im   = ax_heat.imshow(corr, cmap=CMAP_CORR, vmin=-vabs, vmax=vabs,
                          aspect="auto", interpolation="nearest")

    for fi in range(N_FEATURES):
        for ai in range(N_ACTIONS):
            v = corr[fi, ai]
            c = "white" if abs(v) > 0.12 else "#888888"
            ax_heat.text(ai, fi, f"{v:+.2f}", ha="center", va="center",
                         color=c, fontsize=7.5, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Corrélation de Pearson", color=TEXT_COL, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    ax_heat.set_xticks(range(N_ACTIONS))
    ax_heat.set_xticklabels(ACTION_NAMES, color=TEXT_COL, fontsize=9)
    ax_heat.set_yticks(range(N_FEATURES))
    ax_heat.set_yticklabels(FEATURE_NAMES, color=TEXT_COL, fontsize=7.5)
    ax_heat.axhline(y=7.5,  color="#F39C12", linewidth=1.3,
                    linestyle="--", alpha=0.7)
    ax_heat.axhline(y=15.5, color="#9B59B6", linewidth=1.0,
                    linestyle=":", alpha=0.6)
    ax_heat.set_title("Corrélation\nfeature × action",
                      color="white", fontsize=11, fontweight="bold", pad=9)
    for spine in ax_heat.spines.values():
        spine.set_edgecolor(GRID_COL)

    # ── 4 barplots par action ─────────────────────────────────────────────
    positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
    for ai, (row, col_idx) in enumerate(positions):
        ax = fig.add_subplot(gs[row, col_idx])
        ax.set_facecolor(PANEL_BG)
        vals  = corr[:, ai]
        order = np.argsort(np.abs(vals))[::-1]
        ypos  = range(N_FEATURES)
        bar_colors = [ACTION_COLORS[ai] if v >= 0 else "#E74C3C"
                      for v in vals[order]]
        ax.barh(list(ypos), vals[order], color=bar_colors,
                edgecolor="#0D1117", alpha=0.85, height=0.70)
        ax.axvline(x=0, color="#AAAAAA", linewidth=1.0, alpha=0.5)
        ax.set_yticks(list(ypos))
        ax.set_yticklabels([FEATURE_NAMES[i] for i in order],
                           color=TEXT_COL, fontsize=7.5)
        ax.set_xlim(-vabs * 1.2, vabs * 1.2)
        apply_style(ax, xlabel="Corrélation de Pearson")
        ax.set_title(f"Action : {ACTION_NAMES[ai]}",
                     color=ACTION_COLORS[ai], fontsize=11, fontweight="bold", pad=8)
        top_f = order[0]
        ax.text(vals[order][0] * 1.08, 0,
                f"  top: {FEATURE_NAMES[top_f]}",
                va="center", color="#FFD700", fontsize=7)

    plt.savefig(out("xai_dt_correlation.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_dt_correlation.png')}")
    plt.show()

    # ── Bonus : profil sensoriel par action ───────────────────────────────
    _plot_mean_per_action(mean_per_action, std_per_action)


def _plot_mean_per_action(mean_per_action: np.ndarray,
                          std_per_action: np.ndarray):
    fig, axes = plt.subplots(1, N_ACTIONS, figsize=(28, 10), facecolor=BG)
    fig.suptitle(
        "Profil sensoriel par action — Valeur moyenne de chaque feature quand l'agent choisit cette action\n"
        "Barre longue = feature élevée lors de ce choix  |  "
        "Orange pointillé = séparateur catégories",
        fontsize=12, fontweight="bold", color="white", y=1.02
    )

    ypos = np.arange(N_FEATURES)
    label_colors = (["#4FC3F7"] * 8 + ["#81C784"] * 8 +
                    ["#CE93D8"] * 2 + ["#E74C3C"] * 4 + ["#95A5A6"] * 4)

    for ai, ax in enumerate(axes):
        ax.set_facecolor(PANEL_BG)
        means = mean_per_action[ai]
        stds  = std_per_action[ai]

        for row in range(N_FEATURES):
            bg_c = "#0F2233" if row % 2 == 0 else PANEL_BG
            ax.axhspan(row - 0.5, row + 0.5, color=bg_c, alpha=0.5, zorder=0)

        ax.barh(ypos, means, xerr=stds, color=ACTION_COLORS[ai], alpha=0.82,
                edgecolor="#0D1117", height=0.65, zorder=2,
                error_kw=dict(ecolor="#AAAAAA", lw=1, capsize=3))
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + s + 0.005, i, f"{m:.2f}", va="center",
                    ha="left", color=TEXT_COL, fontsize=6.5, alpha=0.85)

        ax.set_yticks(ypos)
        ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
        for tick, col in zip(ax.get_yticklabels(), label_colors):
            tick.set_color(col)

        # Séparateurs catégories
        for ysep, col in [(7.5, "#F39C12"), (15.5, "#9B59B6"),
                          (17.5, "#CE93D8"), (21.5, "#E74C3C")]:
            ax.axhline(y=ysep, color=col, linewidth=1.2,
                       linestyle="--", alpha=0.65, zorder=3)

        ax.set_title(f"Action : {ACTION_NAMES[ai]}",
                     color=ACTION_COLORS[ai], fontsize=12,
                     fontweight="bold", pad=10)
        ax.set_xlabel("Valeur normalisée [0 – 1]", color=TEXT_COL, fontsize=8)
        ax.set_xlim(0, 1.15)
        ax.tick_params(axis="x", colors="#8899AA", labelsize=8)
        ax.tick_params(axis="y", colors="#8899AA", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.5, zorder=1)

    plt.tight_layout()
    plt.savefig(out("xai_dt_mean_per_action.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_dt_mean_per_action.png')}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="XAI – Feature Importance DT Snake"
    )
    parser.add_argument("--permutation", action="store_true",
                        help="Permutation Importance")
    parser.add_argument("--native",      action="store_true",
                        help="Importance native XGBoost (gain/weight/cover)")
    parser.add_argument("--correlation", action="store_true",
                        help="Corrélation features × actions")
    parser.add_argument("--model",    type=str, default="snake_xgb_model.pkl")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Épisodes pour permutation/corrélation (défaut : 20)")
    args = parser.parse_args()

    run_all = not (args.permutation or args.native or args.correlation)

    agent = load_agent(args.model)

    if run_all or args.native:
        print("\n[XAI] ── Importance native ──────────────────────────────")
        plot_native_importance(agent)

    if run_all or args.permutation:
        print(f"\n[XAI] ── Permutation Importance ({args.episodes} épisodes) ──")
        drops, baseline, drops_std = compute_permutation_importance(
            agent, n_episodes=args.episodes
        )
        plot_permutation_importance(drops, baseline, drops_std)

    if run_all or args.correlation:
        print(f"\n[XAI] ── Corrélation features × actions ─────────────────")
        states, actions = collect_states(agent, n_episodes=args.episodes)
        corr, means, stds = compute_correlation(states, actions)
        plot_correlation(corr, means, stds)

    print("\n[XAI] Analyse terminée.")


if __name__ == "__main__":
    main()

# python xai_dt_features.py                         # tout (20 épisodes)
# python xai_dt_features.py --native                 # rapide, pas d'épisodes
# python xai_dt_features.py --permutation --episodes 50
# python xai_dt_features.py --correlation --episodes 30
