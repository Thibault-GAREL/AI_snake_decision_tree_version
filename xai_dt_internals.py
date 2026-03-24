"""
xai_dt_internals.py – Analyse XAI : Structure interne de l'ensemble d'arbres
=============================================================================
Miroir de xai_activations.py pour le modèle XGBoost / GradientBoosting.

3 analyses :
  1. Distributions des importances (gain / weight / cover)
       → analogues des histogrammes d'activations + "features mortes"
         (features jamais utilisées lors des splits)
  2. Features spécialisées par situation de jeu
       → analogue des "neurones spécialisés" : quelles features pilotent
         les décisions selon la situation (danger imminent, food alignée…)
  3. t-SNE / UMAP des états de jeu
       → analogue de la projection 2D des activations :
         clusters de situations similaires, colorés par action / situation

Features (26 valeurs) :
  [0..7]   distances murs   (N NE E SE S SW W NW)
  [8..15]  distances food   (N NE E SE S SW W NW)
  [16]     food_delta_x     (normalisé par largeur)
  [17]     food_delta_y     (normalisé par hauteur)
  [18..21] danger binaire   (N E S W)
  [22..25] direction one-hot(UP RIGHT DOWN LEFT)

Usage :
    python xai_dt_internals.py                   # toutes les analyses
    python xai_dt_internals.py --importance       # distributions d'importance
    python xai_dt_internals.py --specialization   # features spécialisées
    python xai_dt_internals.py --tsne             # t-SNE des états
    python xai_dt_internals.py --umap             # UMAP des états (+ rapide)
    python xai_dt_internals.py --episodes 15      # épisodes de collecte (défaut : 10)
"""

import argparse
import os
import math
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

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
OUT_DIR = "xai_dt_internals"
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

SITUATION_NAMES = [
    "Danger N",      "Danger E",      "Danger S",     "Danger W",
    "Food alignée H","Food alignée V","Food diag.",   "Neutre",
]
SITUATION_COLORS = [
    "#E74C3C","#F39C12","#F1C40F","#2ECC71",
    "#3498DB","#9B59B6","#1ABC9C","#95A5A6",
]

ACTION_NAMES  = ["UP ↑", "RIGHT →", "DOWN ↓", "LEFT ←"]
ACTION_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]

BG       = "#0D1117"
PANEL_BG = "#0D1B2A"
GRID_COL = "#1E3A5F"
TEXT_COL = "#CCDDEE"

CMAP_IMP  = LinearSegmentedColormap.from_list("imp",  ["#0D1B2A","#1F618D","#F39C12","#E74C3C"])
CMAP_SPEC = LinearSegmentedColormap.from_list("spec", ["#0D1B2A","#154360","#1F618D","#D4AC0D","#E74C3C"])
CMAP_TSNE = LinearSegmentedColormap.from_list("tsne", ["#4FC3F7","#81C784","#FFB74D","#F06292"])


# ─────────────────────────────────────────────────────────────────────────────
#  Utilitaires partagés
# ─────────────────────────────────────────────────────────────────────────────
def load_agent(model_path: str = "snake_xgb_model.pkl") -> DecisionTreeAgent:
    agent = DecisionTreeAgent(use_cuda=False)
    agent.load(model_path=model_path)
    if XGBOOST_AVAILABLE and hasattr(agent.model, "get_score"):
        agent.use_cuda = True
    if not agent.trained:
        print("[WARN] Modèle non entraîné — résultats aléatoires.")
    return agent


def predict_proba(agent: DecisionTreeAgent, X: np.ndarray) -> np.ndarray:
    """Retourne [N, 4] probabilités (softmax sur les marges brutes)."""
    X = np.atleast_2d(X).astype(np.float32)
    Xs = agent.scaler.transform(X)
    if XGBOOST_AVAILABLE and hasattr(agent.model, "predict"):
        dmat    = xgb.DMatrix(Xs)
        margins = agent.model.predict(dmat, output_margin=True)   # [N, 4]
        if margins.ndim == 1:
            margins = margins.reshape(-1, N_ACTIONS)
        margins -= margins.max(axis=1, keepdims=True)
        exp_m    = np.exp(margins)
        return exp_m / exp_m.sum(axis=1, keepdims=True)
    else:
        return agent.model.predict_proba(Xs)


def predict_class(agent: DecisionTreeAgent, X: np.ndarray) -> np.ndarray:
    """Retourne [N] classes prédites."""
    X = np.atleast_2d(X).astype(np.float32)
    Xs = agent.scaler.transform(X)
    if XGBOOST_AVAILABLE and hasattr(agent.model, "predict"):
        dmat = xgb.DMatrix(Xs)
        return agent.model.predict(dmat).astype(int)
    else:
        return agent.model.predict(Xs).astype(int)


def _classify_situation(state: list) -> int:
    """Classe l'état en une des 8 situations."""
    d_n  = state[18];  d_e = state[19];  d_s = state[20];  d_w = state[21]
    fdx  = state[16];  fdy = state[17]
    food_h = state[10] + state[14]   # food E + food W
    food_v = state[8]  + state[12]   # food N + food S
    food_diag = (state[9] + state[11] + state[13] + state[15])

    if d_n == 1.0: return 0
    if d_e == 1.0: return 1
    if d_s == 1.0: return 2
    if d_w == 1.0: return 3
    if food_h > 0: return 4
    if food_v > 0: return 5
    if food_diag > 0: return 6
    return 7


class _Collector:
    """Wrapper game_loop — collecte (state, action) à chaque step."""
    def __init__(self, agent):
        self.agent = agent
        self.steps = []

    def tab_state(self, *args):
        return list(args)

    def get_action(self, net, state):
        if len(state) >= 26:
            if   state[22] > 0.5: self.agent.set_direction("UP")
            elif state[23] > 0.5: self.agent.set_direction("RIGHT")
            elif state[24] > 0.5: self.agent.set_direction("DOWN")
            elif state[25] > 0.5: self.agent.set_direction("LEFT")
        action = self.agent.get_action(state)
        self.steps.append((list(state), action))
        return action


def collect_episodes(agent: DecisionTreeAgent, n_episodes: int = 10):
    """
    Joue n_episodes épisodes greedy.
    Retourne (states [T,26], actions [T], situations [T]).
    """
    game.show          = False
    game.stop_iteration = 2000

    all_states, all_actions, all_situations = [], [], []

    for ep in range(n_episodes):
        col = _Collector(agent)
        agent.direction = "RIGHT"
        score = game.game_loop(
            game.rect_width, game.rect_height, game.display,
            agent, None, 0, col
        )
        for state, action in col.steps:
            all_states.append(state)
            all_actions.append(action)
            all_situations.append(_classify_situation(state))
        print(f"  [Collect] Épisode {ep+1}/{n_episodes} → score {score} "
              f"({len(col.steps)} steps | total {len(all_states)})")

    return (
        np.array(all_states,     dtype=np.float32),
        np.array(all_actions,    dtype=np.int32),
        np.array(all_situations, dtype=np.int32),
    )


def apply_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    if title:  ax.set_title(title,  color="white",   fontsize=11, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_COL,  fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_COL,  fontsize=8)
    ax.tick_params(colors="#8899AA", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)


# ─────────────────────────────────────────────────────────────────────────────
#  Analyse 1 – Distributions des importances
# ─────────────────────────────────────────────────────────────────────────────
def _get_importance(agent: DecisionTreeAgent) -> dict:
    """
    Retourne un dict {type: np.ndarray [N_FEATURES]}.
    Pour XGBoost  : gain, weight, cover.
    Pour sklearn  : une seule entrée 'gain'.
    Les features non utilisées reçoivent 0.
    """
    result = {}
    if XGBOOST_AVAILABLE and hasattr(agent.model, "get_score"):
        for itype in ("gain", "weight", "cover"):
            scores_dict = agent.model.get_score(importance_type=itype)
            arr = np.zeros(N_FEATURES, dtype=np.float32)
            for key, val in scores_dict.items():
                fi = int(key[1:])   # 'f7' → 7
                if fi < N_FEATURES:
                    arr[fi] = val
            result[itype] = arr
    else:
        arr = np.array(agent.model.feature_importances_, dtype=np.float32)
        result["gain"] = arr

    return result


def plot_importance_distributions(agent: DecisionTreeAgent):
    """
    Pour chaque type d'importance :
      - Barplot horizontal trié
      - Taux de features "mortes" (jamais utilisées)
    + Heatmap comparative : feature × type d'importance
    """
    imp = _get_importance(agent)
    types = list(imp.keys())
    n_types = len(types)

    # Couleurs par type
    type_colors = {"gain": "#F39C12", "weight": "#4FC3F7", "cover": "#81C784"}
    type_labels = {
        "gain":   "Gain moyen\n(amélioration de pureté)",
        "weight": "Weight\n(nb de splits)",
        "cover":  "Cover\n(nb d'exemples couverts)",
    }

    fig = plt.figure(figsize=(24, 6 + 4 * n_types), facecolor=BG)
    fig.suptitle(
        "Importances des features — Structure interne de l'ensemble d'arbres\n"
        "Gain = amélioration de pureté aux nœuds  |  "
        "Weight = nombre de splits  |  Cover = exemples couverts par splits",
        fontsize=13, fontweight="bold", color="white", y=1.01
    )

    gs = gridspec.GridSpec(n_types + 1, 2, figure=fig,
                           wspace=0.40, hspace=0.55,
                           height_ratios=[4] * n_types + [3],
                           width_ratios=[2, 1.2])

    for row, itype in enumerate(types):
        arr   = imp[itype]
        order = np.argsort(arr)[::-1]
        n_dead = (arr == 0).sum()
        pct_dead = 100 * n_dead / N_FEATURES

        norm_c = Normalize(vmin=0, vmax=arr.max() + 1e-8)
        colors = [CMAP_IMP(norm_c(arr[order[i]])) for i in range(N_FEATURES)]

        # ── Barplot horizontal ─────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 0])
        ax.set_facecolor(PANEL_BG)
        ax.barh(range(N_FEATURES), arr[order], color=colors,
                edgecolor="#0D1117", height=0.72)

        for i, v in enumerate(arr[order]):
            if v > 0:
                ax.text(v * 1.01, i, f"{v:.1f}", va="center",
                        color=TEXT_COL, fontsize=6.5)

        ax.set_yticks(range(N_FEATURES))
        ax.set_yticklabels([FEATURE_NAMES[i] for i in order],
                           color=TEXT_COL, fontsize=7.5)

        # Séparateur murs / food / enrichies
        n_above_mur  = sum(1 for i in order if i < 8)
        n_above_food = sum(1 for i in order if i < 16)
        for yline, label in [
            (N_FEATURES - n_above_mur  - 0.5, "murs ↓"),
            (N_FEATURES - n_above_food - 0.5, "food ↓"),
        ]:
            if 0 < yline < N_FEATURES - 1:
                ax.axhline(y=yline, color="#F39C12", linewidth=1.0,
                           linestyle="--", alpha=0.6)
                ax.text(arr.max() * 0.98, yline + 0.3, label,
                        color="#F39C12", fontsize=7, ha="right", alpha=0.8)

        apply_style(ax, title=f"Importance type : {type_labels.get(itype, itype)}",
                    xlabel=f"Score d'importance ({itype})")
        ax.grid(axis="x", color=GRID_COL, linewidth=0.4, alpha=0.4)

        # Annotation features mortes
        ax.text(0.98, 0.02,
                f"Features mortes (score=0) : {n_dead}/{N_FEATURES} ({pct_dead:.0f}%)",
                transform=ax.transAxes, va="bottom", ha="right",
                color="#E74C3C" if n_dead > 0 else "#2ECC71", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0D1B2A",
                          edgecolor=GRID_COL, alpha=0.9))

        # ── Histogramme des scores ─────────────────────────────────────────
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.set_facecolor(PANEL_BG)
        nonzero = arr[arr > 0]
        ax2.hist(arr, bins=20, color="#1F618D", alpha=0.6,
                 label="Toutes", edgecolor="none")
        if len(nonzero) > 0:
            ax2.hist(nonzero, bins=20,
                     color=type_colors.get(itype, "#F39C12"),
                     alpha=0.8, label="Non-nulles", edgecolor="none")
        ax2.axvline(x=0, color="#E74C3C", linewidth=1.5, linestyle="--",
                    alpha=0.8, label="x = 0 (morte)")
        ax2.set_yscale("log")
        apply_style(ax2,
                    title=f"Distribution des scores ({itype})",
                    xlabel="Score d'importance", ylabel="Fréquence (log)")
        ax2.legend(fontsize=7, facecolor="#0D1117",
                   edgecolor="#444", labelcolor="white")

    # ── Heatmap comparative feature × type ────────────────────────────────
    ax_heat = fig.add_subplot(gs[n_types, :])
    ax_heat.set_facecolor(PANEL_BG)

    # Normalise chaque type en [0,1] pour comparer sur la même échelle
    heat_data = np.zeros((len(types), N_FEATURES))
    for ti, itype in enumerate(types):
        arr = imp[itype]
        mx = arr.max()
        heat_data[ti] = arr / mx if mx > 0 else arr

    # Tri des features par importance gain (ou première)
    feat_order = np.argsort(heat_data[0])[::-1]
    heat_show  = heat_data[:, feat_order]

    im = ax_heat.imshow(heat_show, cmap=CMAP_IMP, vmin=0, vmax=1,
                        aspect="auto", interpolation="nearest")

    ax_heat.set_yticks(range(len(types)))
    ax_heat.set_yticklabels([type_labels.get(t, t).split("\n")[0]
                              for t in types], color=TEXT_COL, fontsize=9)
    ax_heat.set_xticks(range(N_FEATURES))
    ax_heat.set_xticklabels([FEATURE_NAMES[i] for i in feat_order],
                             rotation=40, ha="right", color=TEXT_COL, fontsize=7.5)
    ax_heat.set_title("Heatmap comparative : Importance normalisée [0–1] par type\n"
                      "(features triées par gain décroissant — plus clair = plus important)",
                      color="white", fontsize=11, fontweight="bold", pad=8)
    for spine in ax_heat.spines.values():
        spine.set_edgecolor(GRID_COL)

    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.01, orientation="horizontal")
    cbar.set_label("Importance normalisée", color=TEXT_COL, fontsize=8)
    cbar.ax.xaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color=TEXT_COL)

    plt.savefig(out("xai_dt_importance.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_dt_importance.png')}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Analyse 2 – Features spécialisées par situation
# ─────────────────────────────────────────────────────────────────────────────
def plot_specialization(states: np.ndarray, situations: np.ndarray,
                        actions: np.ndarray):
    """
    Pour chaque situation, calcule la valeur moyenne de chaque feature.
    Score de spécialisation = max_sit(mean_feat) – mean_sit(mean_feat).
    Identifie les features les plus discriminantes par situation.
    """
    n_sit   = len(SITUATION_NAMES)
    n_feat  = N_FEATURES
    sit_arr = situations

    # [F, S] : valeur moyenne de feature fi dans situation si
    means = np.zeros((n_feat, n_sit))
    sit_counts = []
    for si in range(n_sit):
        mask = sit_arr == si
        sit_counts.append(mask.sum())
        if mask.sum() > 0:
            means[:, si] = states[mask].mean(axis=0)

    # Score de spécialisation : max – mean sur les situations
    spec_score = means.max(axis=1) - means.mean(axis=1)
    top_idx    = np.argsort(spec_score)[::-1]

    fig = plt.figure(figsize=(24, 14), facecolor=BG)
    fig.suptitle(
        "Features spécialisées — Quelles features pilotent quelles situations ?\n"
        "Score de spécialisation = max_situation(valeur_moy) – mean_situations(valeur_moy)  "
        "|  Score élevé = feature très sélective d'une situation",
        fontsize=12, fontweight="bold", color="white", y=1.005
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.42, hspace=0.55)

    # ── Distribution des scores de spécialisation ─────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(PANEL_BG)
    ax0.hist(spec_score, bins=20, color="#1F618D", alpha=0.75, edgecolor="none")
    ax0.axvline(x=np.percentile(spec_score, 90), color="#E74C3C",
                linewidth=1.5, linestyle="--", label="90e percentile")
    ax0.axvline(x=np.median(spec_score), color="#F39C12",
                linewidth=1.0, linestyle=":", label="médiane")
    apply_style(ax0, title="Score de spécialisation par feature",
                xlabel="max – mean des valeurs par situation",
                ylabel="Nombre de features")
    ax0.legend(fontsize=7, facecolor="#0D1117",
               edgecolor="#444", labelcolor="white")
    ax0.text(0.97, 0.97,
             f"Top feature : {FEATURE_NAMES[top_idx[0]]}\n"
             f"Score max   : {spec_score[top_idx[0]]:.3f}\n"
             f"Situation   : {SITUATION_NAMES[means[top_idx[0]].argmax()]}",
             transform=ax0.transAxes, va="top", ha="right",
             color="#FFD700", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0D1B2A",
                       edgecolor="#F39C12", alpha=0.9))

    # ── Heatmap [situation × feature] ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.set_facecolor(PANEL_BG)

    top16      = top_idx[:16]
    heat       = means[top16, :].T   # [S, 16]
    vmax_h     = np.percentile(means, 95)
    im = ax1.imshow(heat, cmap=CMAP_SPEC, vmin=0, vmax=max(vmax_h, 1e-6),
                    aspect="auto", interpolation="nearest")

    ax1.set_yticks(range(n_sit))
    ax1.set_yticklabels(
        [f"{SITUATION_NAMES[i]}  (n={sit_counts[i]})" for i in range(n_sit)],
        color=TEXT_COL, fontsize=8.5
    )
    ax1.set_xticks(range(len(top16)))
    ax1.set_xticklabels([FEATURE_NAMES[i] for i in top16],
                        rotation=35, ha="right", color=TEXT_COL, fontsize=8)
    ax1.set_title("Valeur moyenne par situation × feature\n"
                  "(top 16 features les plus spécialisées)",
                  color="white", fontsize=11, fontweight="bold", pad=8)
    for spine in ax1.spines.values():
        spine.set_edgecolor(GRID_COL)
    cbar1 = plt.colorbar(im, ax=ax1, fraction=0.025, pad=0.02)
    cbar1.set_label("Valeur moyenne", color=TEXT_COL, fontsize=8)
    cbar1.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar1.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    # ── Profil des 5 features les plus spécialisées ───────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.set_facecolor(PANEL_BG)
    top5    = top_idx[:5]
    sit_pos = np.arange(n_sit)
    bar_w   = 0.16
    palette = ["#4FC3F7","#81C784","#FFB74D","#F06292","#CE93D8"]
    for rank, fi in enumerate(top5):
        offset = (rank - 2) * bar_w
        ax2.bar(sit_pos + offset, means[fi],
                width=bar_w * 0.9, color=palette[rank], alpha=0.85,
                label=f"{FEATURE_NAMES[fi]} (score={spec_score[fi]:.2f})",
                edgecolor="#0D1117")
    ax2.set_xticks(sit_pos)
    ax2.set_xticklabels(SITUATION_NAMES, rotation=30, ha="right",
                        color=TEXT_COL, fontsize=8)
    ax2.legend(fontsize=7, facecolor="#0D1117",
               edgecolor="#444", labelcolor="white", loc="upper right")
    apply_style(ax2, title="Profil des 5 features les plus spécialisées",
                ylabel="Valeur moyenne de la feature")
    ax2.grid(axis="y", color=GRID_COL, linewidth=0.5, alpha=0.5)

    # ── Actions choisies par situation (distribution) ─────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_facecolor(PANEL_BG)
    action_sit_counts = np.zeros((n_sit, N_ACTIONS))
    for si in range(n_sit):
        mask = sit_arr == si
        if mask.sum() > 0:
            for ai in range(N_ACTIONS):
                action_sit_counts[si, ai] = (actions[mask] == ai).sum()
            tot = action_sit_counts[si].sum()
            if tot > 0:
                action_sit_counts[si] /= tot

    sit_pos2 = np.arange(n_sit)
    bar_w2   = 0.20
    for ai in range(N_ACTIONS):
        offset = (ai - 1.5) * bar_w2
        ax3.bar(sit_pos2 + offset, action_sit_counts[:, ai],
                width=bar_w2 * 0.9, color=ACTION_COLORS[ai],
                alpha=0.85, label=ACTION_NAMES[ai], edgecolor="#0D1117")
    ax3.set_xticks(sit_pos2)
    ax3.set_xticklabels(SITUATION_NAMES, rotation=35, ha="right",
                        color=TEXT_COL, fontsize=7.5)
    ax3.legend(fontsize=7, facecolor="#0D1117",
               edgecolor="#444", labelcolor="white")
    apply_style(ax3, title="Actions choisies par situation\n(fréquence relative)",
                ylabel="Fréquence")
    ax3.grid(axis="y", color=GRID_COL, linewidth=0.5, alpha=0.5)

    plt.savefig(out("xai_dt_specialization.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_dt_specialization.png')}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Analyse 3 – t-SNE / UMAP des états
# ─────────────────────────────────────────────────────────────────────────────
def _run_tsne(data: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    from sklearn.manifold import TSNE
    import sklearn
    from packaging import version
    iter_kwarg = (
        "max_iter" if version.parse(sklearn.__version__) >= version.parse("1.4")
        else "n_iter"
    )
    tsne = TSNE(n_components=2,
                perplexity=min(perplexity, data.shape[0] - 1),
                learning_rate="auto", init="pca", random_state=42,
                **{iter_kwarg: 1000})
    return tsne.fit_transform(data)


def _run_umap(data: np.ndarray) -> np.ndarray:
    try:
        import umap
        return umap.UMAP(n_components=2, n_neighbors=15,
                         min_dist=0.1, random_state=42).fit_transform(data)
    except ImportError:
        print("  [WARN] umap-learn non installé — fallback t-SNE.")
        return _run_tsne(data)


def plot_projection(states: np.ndarray, situations: np.ndarray,
                    actions: np.ndarray, method: str = "tsne"):
    """Projection 2D des états de jeu, colorée par situation / action."""
    method_label = "t-SNE" if method == "tsne" else "UMAP"

    MAX_POINTS = 3000
    T = len(situations)
    idx = (np.random.choice(T, MAX_POINTS, replace=False)
           if T > MAX_POINTS else np.arange(T))
    idx = np.sort(idx)

    print(f"  [{method_label}] {len(idx)} points × {N_FEATURES} dims…")
    data = states[idx]
    proj = _run_tsne(data) if method == "tsne" else _run_umap(data)
    x, y = proj[:, 0], proj[:, 1]
    sits_sub    = situations[idx]
    actions_sub = actions[idx]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor=BG)
    fig.suptitle(
        f"Projection {method_label} des états de jeu — Chaque point = un état\n"
        "Gauche : coloré par situation  |  Droite : coloré par action choisie",
        fontsize=13, fontweight="bold", color="white"
    )

    sit_colors_map = {i: SITUATION_COLORS[i] for i in range(len(SITUATION_NAMES))}
    ALPHA = 0.55
    SIZE  = 8

    # ── Par situation ──────────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.set_facecolor(PANEL_BG)
    for si, sname in enumerate(SITUATION_NAMES):
        mask = sits_sub == si
        if mask.sum() == 0: continue
        ax0.scatter(x[mask], y[mask], c=sit_colors_map[si], s=SIZE,
                    alpha=ALPHA, edgecolors="none",
                    label=f"{sname} ({mask.sum()})")
    ax0.legend(fontsize=7, facecolor="#0D1117", edgecolor="#444",
               labelcolor="white", markerscale=2, loc="best", framealpha=0.85)
    apply_style(ax0,
                title=f"{method_label} — Coloré par situation",
                xlabel=f"{method_label}-1", ylabel=f"{method_label}-2")

    # ── Par action ────────────────────────────────────────────────────────
    ax1 = axes[1]
    ax1.set_facecolor(PANEL_BG)
    for ai, aname in enumerate(ACTION_NAMES):
        mask = actions_sub == ai
        if mask.sum() == 0: continue
        ax1.scatter(x[mask], y[mask], c=ACTION_COLORS[ai], s=SIZE,
                    alpha=ALPHA, edgecolors="none",
                    label=f"{aname} ({mask.sum()})")
    ax1.legend(fontsize=7, facecolor="#0D1117", edgecolor="#444",
               labelcolor="white", markerscale=2, loc="best", framealpha=0.85)
    apply_style(ax1,
                title=f"{method_label} — Coloré par action choisie",
                xlabel=f"{method_label}-1", ylabel=f"{method_label}-2")

    fname = f"xai_dt_{method}.png"
    plt.savefig(out(fname), dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out(fname)}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="XAI – Structure interne de l'ensemble d'arbres Snake"
    )
    parser.add_argument("--importance",     action="store_true",
                        help="Distributions des importances (gain/weight/cover)")
    parser.add_argument("--specialization", action="store_true",
                        help="Features spécialisées par situation de jeu")
    parser.add_argument("--tsne",           action="store_true",
                        help="Projection t-SNE des états")
    parser.add_argument("--umap",           action="store_true",
                        help="Projection UMAP des états")
    parser.add_argument("--episodes",  type=int, default=10,
                        help="Épisodes de collecte (défaut : 10)")
    parser.add_argument("--model",     type=str, default="snake_xgb_model.pkl",
                        help="Chemin du fichier modèle (défaut : snake_xgb_model.pkl)")
    args = parser.parse_args()

    run_all = not (args.importance or args.specialization
                   or args.tsne or args.umap)

    agent = load_agent(args.model)

    # Les analyses 2/3 nécessitent des épisodes
    if run_all or args.specialization or args.tsne or args.umap:
        print(f"\n[XAI] Collecte sur {args.episodes} épisode(s)…")
        states, actions, situations = collect_episodes(agent, n_episodes=args.episodes)
        print(f"[XAI] {len(states)} steps collectés.\n")
    else:
        states = actions = situations = None

    if run_all or args.importance:
        print("[XAI] ── Distributions des importances ──────────────────")
        plot_importance_distributions(agent)

    if run_all or args.specialization:
        print("[XAI] ── Features spécialisées ──────────────────────────")
        plot_specialization(states, situations, actions)

    if run_all or args.tsne:
        print("[XAI] ── t-SNE des états ─────────────────────────────────")
        try:
            from sklearn.manifold import TSNE
            plot_projection(states, situations, actions, method="tsne")
        except ImportError:
            print("  [WARN] scikit-learn non installé.")

    if run_all or args.umap:
        print("[XAI] ── UMAP des états ──────────────────────────────────")
        plot_projection(states, situations, actions, method="umap")

    print("\n[XAI] Analyse terminée.")


if __name__ == "__main__":
    main()

# python xai_dt_internals.py                        # tout (10 épisodes)
# python xai_dt_internals.py --importance            # rapide, pas d'épisodes
# python xai_dt_internals.py --specialization --episodes 20
# python xai_dt_internals.py --tsne --episodes 20
# python xai_dt_internals.py --umap --episodes 20
