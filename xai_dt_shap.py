"""
xai_dt_shap.py – Analyse XAI : SHAP pour le modèle Snake (XGBoost / sklearn)
==============================================================================
Miroir de xai_shap.py pour le modèle XGBoost / GradientBoosting.

Utilise le SHAP natif de XGBoost (pred_contribs=True, sans DeepExplainer)
ou shap.TreeExplainer pour sklearn — beaucoup plus rapide que DeepExplainer.

4 visualisations :
  1. Beeswarm plot  — vue globale : impact de chaque feature sur l'ensemble
                      des décisions (N états collectés)
  2. Waterfall plot — vue locale  : décomposition d'une décision représentative
                      par situation de jeu
  3. Force plot     — vue locale HTML interactif (contributions cumulatives)
  4. Summary heatmap — matrice SHAP [feature × action] et [feature × situation]

Installation recommandée :
    pip install shap --break-system-packages

Usage :
    python xai_dt_shap.py                     # toutes les visualisations
    python xai_dt_shap.py --beeswarm          # beeswarm global
    python xai_dt_shap.py --waterfall         # waterfall par situation
    python xai_dt_shap.py --force             # force plots HTML
    python xai_dt_shap.py --heatmap           # summary heatmap
    python xai_dt_shap.py --episodes 15       # épisodes de collecte (défaut : 12)
    python xai_dt_shap.py --model snake_xgb_model.pkl
"""

import argparse
import os
import math
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
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
OUT_DIR = "xai_dt_shap"
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

CMAP_SHAP = LinearSegmentedColormap.from_list(
    "shap_div", ["#C0392B","#E8A090","#F5F5F5","#90C8E8","#1A5276"]
)
CMAP_ABS = LinearSegmentedColormap.from_list(
    "shap_abs", ["#0D1B2A","#154360","#1F618D","#F39C12","#E74C3C"]
)


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


def _classify_situation(state: list) -> int:
    d_n = state[18]; d_e = state[19]; d_s = state[20]; d_w = state[21]
    food_h    = state[10] + state[14]
    food_v    = state[8]  + state[12]
    food_diag = state[9]  + state[11] + state[13] + state[15]
    if d_n == 1.0: return 0
    if d_e == 1.0: return 1
    if d_s == 1.0: return 2
    if d_w == 1.0: return 3
    if food_h > 0: return 4
    if food_v > 0: return 5
    if food_diag > 0: return 6
    return 7


class _Collector:
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


def collect_states(agent: DecisionTreeAgent, n_episodes: int = 12):
    """
    Joue n_episodes épisodes greedy.
    Retourne (states [T,26], actions [T], situations [T]).
    """
    game.show           = False
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
              f"| total steps : {len(all_states)}")

    return (
        np.array(all_states,     dtype=np.float32),
        np.array(all_actions,    dtype=np.int32),
        np.array(all_situations, dtype=np.int32),
    )


def apply_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    if title:  ax.set_title(title,  color="white",  fontsize=11, fontweight="bold", pad=9)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
    ax.tick_params(colors="#8899AA", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)


# ─────────────────────────────────────────────────────────────────────────────
#  Calcul des valeurs SHAP
# ─────────────────────────────────────────────────────────────────────────────
def compute_shap_values(agent: DecisionTreeAgent,
                        states: np.ndarray) -> tuple:
    """
    Calcule les valeurs SHAP :
      - XGBoost natif  : pred_contribs=True  (rapide, exact pour les arbres)
      - sklearn        : shap.TreeExplainer

    Retourne :
        shap_values : list[np.ndarray]  — un array [T, N_FEATURES] par action
        expected    : np.ndarray        — E[f(x)] par action (valeur de base)
    """
    Xs = agent.scaler.transform(states.astype(np.float32))

    if XGBOOST_AVAILABLE and hasattr(agent.model, "predict"):
        # ── XGBoost natif ─────────────────────────────────────────────────
        print(f"  [SHAP] XGBoost natif (pred_contribs) sur {len(states)} états…")
        dmat  = xgb.DMatrix(Xs)
        raw   = agent.model.predict(dmat, pred_contribs=True)   # [T, K*(F+1)]
        T     = len(states)
        F     = N_FEATURES
        K     = N_ACTIONS

        # Reshape : [T, K, F+1]  (dernière colonne = bias par classe)
        raw = raw.reshape(T, K, F + 1)
        shap_values = [raw[:, k, :F] for k in range(K)]   # liste de K arrays [T, F]
        expected    = raw[0, :, F]                          # [K] — base value (constante)

        print(f"  [SHAP] ✓ Shape par action : {shap_values[0].shape}")
        return shap_values, expected

    else:
        # ── Sklearn TreeExplainer ─────────────────────────────────────────
        try:
            import shap
        except ImportError:
            raise ImportError(
                "shap non installé.\n"
                "pip install shap --break-system-packages"
            )
        print(f"  [SHAP] shap.TreeExplainer (sklearn) sur {len(states)} états…")
        explainer = shap.TreeExplainer(agent.model)
        sv        = explainer.shap_values(Xs)   # list [K] de [T, F] ou [T, F, K]

        if isinstance(sv, list):
            shap_values = [np.array(v) for v in sv]
        else:
            sv = np.array(sv)
            if sv.ndim == 3:   # [T, F, K]
                shap_values = [sv[:, :, k] for k in range(K)]
            else:
                shap_values = [sv] * K

        expected = explainer.expected_value
        if not hasattr(expected, "__len__"):
            expected = np.full(K, float(expected))
        expected = np.array(expected)

        print(f"  [SHAP] ✓ Shape par action : {shap_values[0].shape}")
        return shap_values, expected


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation 1 – Beeswarm plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_beeswarm(shap_values: list, states: np.ndarray):
    """
    Beeswarm plot — 4 subplots (un par action).
    Chaque point = un état. Axe X = valeur SHAP (impact sur la marge brute).
    Couleur = valeur de la feature (froid = faible, chaud = élevé).

    Lecture :
      - Features en haut = plus d'impact global
      - Point à droite   = feature pousse VERS cette action
      - Point à gauche   = feature freine cette action
    """
    fig, axes = plt.subplots(1, N_ACTIONS, figsize=(28, 10), facecolor=BG)
    fig.suptitle(
        "SHAP Beeswarm — Impact de chaque feature sur chaque action\n"
        "Chaque point = un état  |  Axe X : valeur SHAP (+→ pousse vers l'action, –→ freine)  |  "
        "Couleur : valeur normalisée de la feature (froid=faible, chaud=élevé)",
        fontsize=12, fontweight="bold", color="white", y=1.02
    )

    mean_abs_all = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    feat_order   = np.argsort(mean_abs_all)   # croissant → top en haut du barh

    CMAP_FEAT = matplotlib.colormaps.get_cmap("coolwarm")

    for ai, ax in enumerate(axes):
        ax.set_facecolor(PANEL_BG)
        sv = shap_values[ai]   # [T, F]
        T  = sv.shape[0]

        for rank, fi in enumerate(feat_order):
            shap_fi = sv[:, fi]
            feat_fi = states[:, fi]

            jitter = np.random.uniform(-0.35, 0.35, size=T)
            y_vals = rank + jitter

            f_min, f_max = feat_fi.min(), feat_fi.max()
            feat_norm = (feat_fi - f_min) / (f_max - f_min + 1e-8)

            ax.scatter(shap_fi, y_vals, c=feat_norm, cmap=CMAP_FEAT,
                       s=6, alpha=0.55, edgecolors="none", vmin=0, vmax=1)

        ax.axvline(x=0, color="#AAAAAA", linewidth=1.0,
                   linestyle="--", alpha=0.6)
        ax.set_yticks(range(N_FEATURES))
        ax.set_yticklabels([FEATURE_NAMES[fi] for fi in feat_order],
                           color=TEXT_COL, fontsize=7.5)

        # Séparateur murs / food
        n_mur = sum(1 for fi in feat_order if fi < 8)
        ax.axhline(y=n_mur - 0.5, color="#F39C12", linewidth=1.2,
                   linestyle=":", alpha=0.7)

        apply_style(ax, xlabel="Valeur SHAP (impact sur la marge)")
        ax.set_title(f"Action : {ACTION_NAMES[ai]}",
                     color=ACTION_COLORS[ai], fontsize=11,
                     fontweight="bold", pad=9)
        ax.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.5)

    # Colorbar globale
    sm = ScalarMappable(cmap=CMAP_FEAT, norm=Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.008, 0.65])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Valeur de la feature (normalisée)",
                   color=TEXT_COL, fontsize=9, rotation=270, labelpad=14)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["Faible", "Moyen", "Élevé"])

    plt.subplots_adjust(right=0.90, wspace=0.45)
    plt.savefig(out("xai_dt_shap_beeswarm.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_dt_shap_beeswarm.png')}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation 2 – Waterfall plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_waterfall(shap_values: list, states: np.ndarray,
                   situations: np.ndarray, expected: np.ndarray):
    """
    Pour chaque situation de jeu (8 situations), sélectionne l'état le plus
    représentatif et trace un waterfall plot pour l'action dominante.

    Waterfall : part de E[f(x)] (base), ajoute les contributions feature par
    feature → arrive à f(x) (marge brute prédite).
    Bleu = contribution positive, rouge = contribution négative.
    """
    n_sit  = len(SITUATION_NAMES)
    n_cols = 4
    n_rows = math.ceil(n_sit / n_cols)

    fig = plt.figure(figsize=(26, 6.5 * n_rows), facecolor=BG)
    fig.suptitle(
        "SHAP Waterfall — Décomposition d'une décision représentative par situation\n"
        "Départ = E[f(x)] (marge de base)  →  Arrivée = marge brute prédite  |  "
        "Bleu = contribution + , Rouge = contribution –",
        fontsize=12, fontweight="bold", color="white", y=1.01
    )

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           wspace=0.48, hspace=0.68)

    from collections import Counter

    for si in range(n_sit):
        row = si // n_cols
        col = si  % n_cols
        ax  = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PANEL_BG)

        mask = situations == si
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        indices = np.where(mask)[0]

        # Action dominante dans cette situation
        action_counts = Counter()
        for idx in indices:
            sv_all = np.array([shap_values[ai][idx] for ai in range(N_ACTIONS)])
            action_counts[int(sv_all.sum(axis=1).argmax())] += 1
        dominant_action = action_counts.most_common(1)[0][0]

        sv_action = shap_values[dominant_action]   # [T, F]
        total_shap = np.abs(sv_action[indices]).sum(axis=1)
        median_val  = np.median(total_shap)
        rep_idx     = indices[np.argmin(np.abs(total_shap - median_val))]

        shap_rep  = sv_action[rep_idx]       # [F]
        state_rep = states[rep_idx]           # [F]
        base_val  = (float(expected[dominant_action])
                     if hasattr(expected, "__len__") else float(expected))

        # Tri par |SHAP| décroissant
        order      = np.argsort(np.abs(shap_rep))[::-1]
        feat_vals  = state_rep[order]
        shap_ord   = shap_rep[order]

        # Positions cumulatives
        cumulative    = np.zeros(len(order) + 1)
        cumulative[0] = base_val
        for k, s in enumerate(shap_ord):
            cumulative[k + 1] = cumulative[k] + s
        final_val = cumulative[-1]

        bar_bottoms = cumulative[:-1].copy()
        bar_heights = shap_ord.copy()
        for k in range(len(shap_ord)):
            if shap_ord[k] < 0:
                bar_bottoms[k] = cumulative[k + 1]
                bar_heights[k] = -shap_ord[k]

        colors_wf = ["#2E86C1" if s >= 0 else "#C0392B" for s in shap_ord]

        ax.barh(range(len(order)), bar_heights, left=bar_bottoms,
                color=colors_wf, edgecolor="#0D1117", height=0.68, alpha=0.88)

        for k, (b, h, s) in enumerate(zip(bar_bottoms, bar_heights, shap_ord)):
            x_txt = b + h + (0.005 if s >= 0 else -0.005)
            ha    = "left" if s >= 0 else "right"
            ax.text(x_txt, k, f"{s:+.3f}", va="center", ha=ha, fontsize=6,
                    color="#AADDFF" if s >= 0 else "#FFAAAA")

        ax.axvline(x=base_val,  color="#F39C12", linewidth=1.2,
                   linestyle="--", alpha=0.8, label=f"E[f]={base_val:.2f}")
        ax.axvline(x=final_val, color="#2ECC71", linewidth=1.4,
                   linestyle="-",  alpha=0.8, label=f"f(x)={final_val:.2f}")

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(
            [f"{FEATURE_NAMES[order[k]]}  [{feat_vals[k]:.2f}]"
             for k in range(len(order))],
            fontsize=6.5, color=TEXT_COL
        )

        ax.set_title(
            f"{SITUATION_NAMES[si]}  →  {ACTION_NAMES[dominant_action]}",
            color=SITUATION_COLORS[si], fontsize=10, fontweight="bold", pad=7
        )
        ax.legend(fontsize=6, facecolor="#0D1117",
                  edgecolor="#444", labelcolor="white", loc="lower right")

        pos_p = mpatches.Patch(color="#2E86C1", label="Contribution +")
        neg_p = mpatches.Patch(color="#C0392B", label="Contribution –")
        ax.legend(handles=[pos_p, neg_p], fontsize=6, facecolor="#0D1117",
                  edgecolor="#444", labelcolor="white", loc="lower right")

        ax.set_xlabel("Marge brute (contributions cumulées)", color=TEXT_COL, fontsize=7)
        ax.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.tick_params(colors="#8899AA", labelsize=7)

    plt.savefig(out("xai_dt_shap_waterfall.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_dt_shap_waterfall.png')}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation 3 – Force plot (HTML)
# ─────────────────────────────────────────────────────────────────────────────
def plot_force(shap_values: list, states: np.ndarray,
               situations: np.ndarray, expected: np.ndarray):
    """
    Génère un force plot HTML interactif (via shap) par situation + global.
    Le force plot montre comment les features « poussent » ou « freinent »
    la marge brute par rapport à la valeur de base E[f(x)].
    """
    try:
        import shap
    except ImportError:
        print("[SKIP] shap non installé — force plot ignoré.")
        return

    shap.initjs()
    from collections import Counter

    for si in range(len(SITUATION_NAMES)):
        mask    = situations == si
        if mask.sum() == 0:
            continue
        indices = np.where(mask)[0][:50]

        action_counts = Counter()
        for idx in indices:
            sv_all = np.array([shap_values[ai][idx] for ai in range(N_ACTIONS)])
            action_counts[int(sv_all.sum(axis=1).argmax())] += 1
        dominant_action = action_counts.most_common(1)[0][0]

        sv_sit  = shap_values[dominant_action][indices]
        st_sit  = states[indices]
        base_v  = (float(expected[dominant_action])
                   if hasattr(expected, "__len__") else float(expected))

        html_path = out(
            f"xai_dt_force_sit{si}_{SITUATION_NAMES[si].replace(' ','_')}.html"
        )
        try:
            fp = shap.force_plot(
                base_value=base_v, shap_values=sv_sit,
                features=st_sit, feature_names=FEATURE_NAMES,
                show=False, matplotlib=False,
            )
            shap.save_html(html_path, fp)
            print(f"[XAI] Sauvegarde → {html_path}")
        except Exception as e:
            print(f"[WARN] Force plot situation {si} : {e}")

    # Force plot global
    from collections import Counter
    all_best = []
    for i in range(len(situations)):
        sv_all = np.array([shap_values[ai][i] for ai in range(N_ACTIONS)])
        all_best.append(int(sv_all.sum(axis=1).argmax()))
    global_dom = Counter(all_best).most_common(1)[0][0]
    sv_global  = shap_values[global_dom]
    base_global = (float(expected[global_dom])
                   if hasattr(expected, "__len__") else float(expected))

    MAX_HTML = 500
    idx_html = np.linspace(0, len(situations) - 1,
                           min(MAX_HTML, len(situations)), dtype=int)
    html_global = out("xai_dt_force_global.html")
    try:
        fp_global = shap.force_plot(
            base_value=base_global,
            shap_values=sv_global[idx_html],
            features=states[idx_html],
            feature_names=FEATURE_NAMES,
            show=False, matplotlib=False,
        )
        shap.save_html(html_global, fp_global)
        print(f"[XAI] Sauvegarde → {html_global}")
    except Exception as e:
        print(f"[WARN] Force plot global : {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation 4 – Summary heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary_heatmap(shap_values: list, states: np.ndarray,
                          situations: np.ndarray):
    """
    4 sous-figures :
      A) |SHAP| moyen [feature × action]       — importance absolue
      B) SHAP signé moyen [feature × action]   — direction de l'influence
      C) Ranking global d'importance (barplot)
      D) |SHAP| moyen [feature × situation]    — importance par contexte de jeu
    """
    mean_abs_matrix  = np.zeros((N_FEATURES, N_ACTIONS))
    mean_sign_matrix = np.zeros((N_FEATURES, N_ACTIONS))
    for ai in range(N_ACTIONS):
        mean_abs_matrix[:, ai]  = np.abs(shap_values[ai]).mean(axis=0)
        mean_sign_matrix[:, ai] = shap_values[ai].mean(axis=0)

    global_importance = mean_abs_matrix.mean(axis=1)
    feat_order        = np.argsort(global_importance)   # croissant

    mean_sit_matrix = np.zeros((N_FEATURES, len(SITUATION_NAMES)))
    for si in range(len(SITUATION_NAMES)):
        mask = situations == si
        if mask.sum() == 0:
            continue
        for ai in range(N_ACTIONS):
            mean_sit_matrix[:, si] += np.abs(shap_values[ai][mask]).mean(axis=0)
        mean_sit_matrix[:, si] /= N_ACTIONS

    fig = plt.figure(figsize=(26, 16), facecolor=BG)
    fig.suptitle(
        "SHAP Summary — Vue globale de l'importance des features\n"
        "Calculé sur l'ensemble des états collectés  |  "
        "Séparateur orange = murs / food  |  Violet pointillé = food / enrichies",
        fontsize=13, fontweight="bold", color="white"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.42, hspace=0.52)

    # ── A) |SHAP| moyen [F × A] ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(PANEL_BG)
    data_a = mean_abs_matrix[feat_order, :]
    vmax_a = np.percentile(data_a, 97)
    im_a   = ax_a.imshow(data_a, cmap=CMAP_ABS, vmin=0,
                          vmax=max(vmax_a, 1e-6), aspect="auto",
                          interpolation="nearest")
    for fi_r, fi in enumerate(feat_order):
        for ai in range(N_ACTIONS):
            v = mean_abs_matrix[fi, ai]
            c = "white" if v > vmax_a * 0.5 else TEXT_COL
            ax_a.text(ai, fi_r, f"{v:.3f}", ha="center", va="center",
                      color=c, fontsize=7.5)
    ax_a.set_xticks(range(N_ACTIONS))
    ax_a.set_xticklabels(ACTION_NAMES, color=TEXT_COL, fontsize=9)
    ax_a.set_yticks(range(N_FEATURES))
    ax_a.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8)
    n_mur_a = sum(1 for i in feat_order if i < 8)
    ax_a.axhline(y=n_mur_a - 0.5, color="#F39C12", linewidth=1.2,
                 linestyle="--", alpha=0.7)
    cbar_a = plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cbar_a.set_label("|SHAP| moyen", color=TEXT_COL, fontsize=8)
    cbar_a.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar_a.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    apply_style(ax_a, title="|SHAP| moyen par feature × action\n"
                             "(importance absolue — plus clair = plus impactant)")

    # ── B) SHAP signé [F × A] ────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(PANEL_BG)
    data_b = mean_sign_matrix[feat_order, :]
    vabs_b = max(np.abs(data_b).max(), 1e-8)
    norm_b = TwoSlopeNorm(vcenter=0, vmin=-vabs_b, vmax=vabs_b)
    im_b   = ax_b.imshow(data_b, cmap=CMAP_SHAP, norm=norm_b,
                          aspect="auto", interpolation="nearest")
    for fi_r, fi in enumerate(feat_order):
        for ai in range(N_ACTIONS):
            v   = mean_sign_matrix[fi, ai]
            col = "white" if abs(v) > vabs_b * 0.35 else TEXT_COL
            ax_b.text(ai, fi_r, f"{v:+.3f}", ha="center", va="center",
                      color=col, fontsize=7.5)
    ax_b.set_xticks(range(N_ACTIONS))
    ax_b.set_xticklabels(ACTION_NAMES, color=TEXT_COL, fontsize=9)
    ax_b.set_yticks(range(N_FEATURES))
    ax_b.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8)
    ax_b.axhline(y=n_mur_a - 0.5, color="#F39C12", linewidth=1.2,
                 linestyle="--", alpha=0.7)
    cbar_b = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cbar_b.set_label("SHAP signé moyen", color=TEXT_COL, fontsize=8)
    cbar_b.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar_b.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    apply_style(ax_b, title="SHAP signé moyen par feature × action\n"
                             "(bleu = influence +, rouge = influence –)")

    # ── C) Barplot importance globale ─────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor(PANEL_BG)
    gi_sorted  = global_importance[feat_order]
    norm_c     = Normalize(vmin=0, vmax=max(gi_sorted.max(), 1e-8))
    colors_c   = [CMAP_ABS(norm_c(v)) for v in gi_sorted]
    ax_c.barh(range(N_FEATURES), gi_sorted, color=colors_c,
              edgecolor="#0D1117", height=0.72)
    for row_idx in range(N_FEATURES):
        bg = "#0F2233" if row_idx % 2 == 0 else PANEL_BG
        ax_c.axhspan(row_idx - 0.5, row_idx + 0.5, color=bg, alpha=0.4, zorder=0)
    for k, v in enumerate(gi_sorted):
        ax_c.text(v + 0.0003, k, f"{v:.4f}", va="center", color=TEXT_COL, fontsize=7.5)
    ax_c.set_yticks(range(N_FEATURES))
    ax_c.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8.5)
    ax_c.axhline(y=n_mur_a - 0.5, color="#F39C12",
                 linewidth=1.2, linestyle="--", alpha=0.7)
    n_food = sum(1 for i in feat_order if 8 <= i < 16)
    ax_c.text(gi_sorted.max() * 0.98, n_mur_a / 2 - 0.5,
              "MURS", color=ACTION_COLORS[0], fontsize=8,
              fontweight="bold", va="center", ha="right", alpha=0.7)
    apply_style(ax_c,
                title="Importance SHAP globale (toutes actions)\n"
                      "Rang ↑ = feature la plus influente",
                xlabel="|SHAP| moyen (toutes actions)")
    ax_c.grid(axis="x", color=GRID_COL, linewidth=0.5, alpha=0.5)

    # ── D) |SHAP| moyen [F × situation] ──────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(PANEL_BG)
    data_d = mean_sit_matrix[feat_order, :]
    vmax_d = np.percentile(data_d, 97)
    im_d   = ax_d.imshow(data_d, cmap=CMAP_ABS, vmin=0,
                          vmax=max(vmax_d, 1e-6), aspect="auto",
                          interpolation="nearest")
    ax_d.set_xticks(range(len(SITUATION_NAMES)))
    ax_d.set_xticklabels(
        [s.replace(" ", "\n") for s in SITUATION_NAMES],
        color=TEXT_COL, fontsize=7.5
    )
    ax_d.set_yticks(range(N_FEATURES))
    ax_d.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8)
    ax_d.axhline(y=n_mur_a - 0.5, color="#F39C12",
                 linewidth=1.2, linestyle="--", alpha=0.7)
    for si, col in enumerate(SITUATION_COLORS):
        ax_d.axvline(x=si - 0.5, color=col, linewidth=0.6, alpha=0.4)
    cbar_d = plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cbar_d.set_label("|SHAP| moyen", color=TEXT_COL, fontsize=8)
    cbar_d.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar_d.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    apply_style(ax_d, title="|SHAP| moyen par feature × situation de jeu\n"
                             "(quelle feature devient cruciale dans quelle situation ?)")

    plt.savefig(out("xai_dt_shap_heatmap.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_dt_shap_heatmap.png')}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="XAI – SHAP pour DT Snake"
    )
    parser.add_argument("--beeswarm",  action="store_true",
                        help="Beeswarm plot global")
    parser.add_argument("--waterfall", action="store_true",
                        help="Waterfall plot par situation")
    parser.add_argument("--force",     action="store_true",
                        help="Force plots HTML interactifs")
    parser.add_argument("--heatmap",   action="store_true",
                        help="Summary heatmap feature × action / situation")
    parser.add_argument("--model",     type=str, default="snake_xgb_model.pkl")
    parser.add_argument("--episodes",  type=int, default=12,
                        help="Épisodes de collecte (défaut : 12)")
    args = parser.parse_args()

    run_all = not (args.beeswarm or args.waterfall
                   or args.force or args.heatmap)

    # ── Chargement & collecte ────────────────────────────────────────────
    agent = load_agent(args.model)

    print(f"\n[XAI] Collecte sur {args.episodes} épisode(s)…")
    states, actions, situations = collect_states(
        agent, n_episodes=args.episodes
    )
    print(f"[XAI] {len(states)} états collectés.\n")

    # ── Calcul SHAP ──────────────────────────────────────────────────────
    print("[XAI] Calcul des valeurs SHAP…")
    shap_values, expected = compute_shap_values(agent, states)

    # ── Visualisations ───────────────────────────────────────────────────
    if run_all or args.beeswarm:
        print("\n[XAI] ── Beeswarm plot ──────────────────────────────────")
        plot_beeswarm(shap_values, states)

    if run_all or args.waterfall:
        print("\n[XAI] ── Waterfall plot ─────────────────────────────────")
        plot_waterfall(shap_values, states, situations, expected)

    if run_all or args.heatmap:
        print("\n[XAI] ── Summary heatmap ────────────────────────────────")
        plot_summary_heatmap(shap_values, states, situations)

    if run_all or args.force:
        print("\n[XAI] ── Force plots (HTML) ─────────────────────────────")
        try:
            import shap
            plot_force(shap_values, states, situations, expected)
        except ImportError:
            print("  [SKIP] shap non installé.")
            print("         pip install shap --break-system-packages")

    print(f"\n[XAI] Analyse SHAP terminée. Fichiers dans : {OUT_DIR}/")


if __name__ == "__main__":
    main()

# python xai_dt_shap.py                              # tout (12 épisodes)
# python xai_dt_shap.py --beeswarm                   # le plus informatif
# python xai_dt_shap.py --waterfall
# python xai_dt_shap.py --heatmap
# python xai_dt_shap.py --force                      # génère les HTML
# python xai_dt_shap.py --episodes 25                # plus de données
# python xai_dt_shap.py --beeswarm --heatmap         # combinaison
