"""
xai_dt_predictions.py – Analyse XAI : Prédictions et confiance du modèle
=========================================================================
Miroir de xai_qvalues.py pour le modèle XGBoost / GradientBoosting.

3 visualisations :
  1. Heatmaps des probabilités prédites des 4 actions sur la grille
     (position de nourriture fixée) — analogue des heatmaps Q-values
  2. Carte de confiance (prob_max – prob_2nd_max) + politique apprise
     — analogue du Q-gap
  3. Évolution temporelle des probabilités pendant un épisode complet
     — analogue de l'évolution temporelle des Q-values

Features (26 valeurs) :
  [0..7]   distances murs   (N NE E SE S SW W NW)
  [8..15]  distances food   (N NE E SE S SW W NW)
  [16]     food_delta_x     [17] food_delta_y
  [18..21] danger binaire   (N E S W)
  [22..25] direction one-hot(UP RIGHT DOWN LEFT)

Usage :
    python xai_dt_predictions.py                 # toutes les visualisations
    python xai_dt_predictions.py --heatmap        # heatmaps de probabilité
    python xai_dt_predictions.py --gap            # carte de confiance + politique
    python xai_dt_predictions.py --temporal       # évolution temporelle
    python xai_dt_predictions.py --food-col 5 --food-row 3
    python xai_dt_predictions.py --episodes 3     # épisodes temporels (défaut : 3)
    python xai_dt_predictions.py --model snake_xgb_model.pkl
"""

import argparse
import math
import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
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
OUT_DIR = "xai_dt_predictions"
os.makedirs(OUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    return os.path.join(OUT_DIR, filename)


# ─────────────────────────────────────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────────────────────────────────────
GRID_W = int(game.width   // game.rect_width)    # 16 colonnes
GRID_H = int(game.height  // game.rect_height)   # 8 lignes

ACTION_NAMES   = ["UP ↑", "RIGHT →", "DOWN ↓", "LEFT ←"]
ACTION_COLORS  = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]

CMAP_PROB = LinearSegmentedColormap.from_list(
    "prob", ["#0D1B2A","#1B4F72","#2E86C1","#F39C12","#E74C3C","#FFFFFF"]
)
CMAP_GAP = LinearSegmentedColormap.from_list(
    "gap", ["#1A1A2E","#16213E","#0F3460","#533483","#E94560"]
)

BG       = "#0D1117"
PANEL_BG = "#0D1B2A"
GRID_COL = "#1E3A5F"
TEXT_COL = "#CCDDEE"


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
    """Retourne [N, 4] probabilités."""
    X = np.atleast_2d(X).astype(np.float32)
    Xs = agent.scaler.transform(X)
    if XGBOOST_AVAILABLE and hasattr(agent.model, "predict"):
        dmat    = xgb.DMatrix(Xs)
        margins = agent.model.predict(dmat, output_margin=True)
        if margins.ndim == 1:
            margins = margins.reshape(-1, N_ACTIONS)
        margins -= margins.max(axis=1, keepdims=True)
        exp_m    = np.exp(margins)
        return exp_m / exp_m.sum(axis=1, keepdims=True)
    else:
        return agent.model.predict_proba(Xs)


def build_state_at(col: int, row: int,
                   food_col: int, food_row: int,
                   direction: str = "RIGHT") -> list:
    """Construit les 26 features pour une position (col, row) donnée."""
    tmp_snake = game.Manager_snake()
    tmp_snake.add_snake(
        game.Snake(col * game.rect_width, row * game.rect_height)
    )
    tmp_snake.direction = direction
    tmp_food = game.food(food_col * game.rect_width, food_row * game.rect_height)

    hx = col      * game.rect_width
    hy = row      * game.rect_height
    fx = food_col * game.rect_width
    fy = food_row * game.rect_height

    return [
        game.distance_bord_north(tmp_snake),
        game.distance_bord_north_est(tmp_snake),
        game.distance_bord_est(tmp_snake),
        game.distance_bord_south_est(tmp_snake),
        game.distance_bord_south(tmp_snake),
        game.distance_bord_south_west(tmp_snake),
        game.distance_bord_west(tmp_snake),
        game.distance_bord_north_west(tmp_snake),
        game.distance_food_north(tmp_snake, tmp_food),
        game.distance_food_north_est(tmp_snake, tmp_food),
        game.distance_food_est(tmp_snake, tmp_food),
        game.distance_food_south_est(tmp_snake, tmp_food),
        game.distance_food_south(tmp_snake, tmp_food),
        game.distance_food_south_west(tmp_snake, tmp_food),
        game.distance_food_west(tmp_snake, tmp_food),
        game.distance_food_north_west(tmp_snake, tmp_food),
        (fx - hx) / game.width,
        (fy - hy) / game.height,
        game.danger_north(tmp_snake),
        game.danger_east(tmp_snake),
        game.danger_south(tmp_snake),
        game.danger_west(tmp_snake),
        1.0 if direction == "UP"    else 0.0,
        1.0 if direction == "RIGHT" else 0.0,
        1.0 if direction == "DOWN"  else 0.0,
        1.0 if direction == "LEFT"  else 0.0,
    ]


def scan_grid(agent: DecisionTreeAgent,
              food_col: int, food_row: int,
              direction: str = "RIGHT") -> tuple:
    """
    Parcourt toutes les cellules, calcule les probabilités prédites.

    Retourne :
        prob_map : [GRID_H, GRID_W, 4]  — probabilité par action
        best     : [GRID_H, GRID_W]     — action choisie (argmax)
        gap      : [GRID_H, GRID_W]     — prob_max – prob_2nd_max (confiance)
    """
    prob_map = np.zeros((GRID_H, GRID_W, N_ACTIONS), dtype=np.float32)

    for row in range(GRID_H):
        for col in range(GRID_W):
            state = build_state_at(col, row, food_col, food_row, direction)
            prob  = predict_proba(agent, np.array(state))[0]
            prob_map[row, col] = prob

    sorted_p = np.sort(prob_map, axis=2)
    best     = np.argmax(prob_map, axis=2)
    gap      = sorted_p[:, :, -1] - sorted_p[:, :, -2]

    return prob_map, best, gap


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation 1 – Heatmaps de probabilité
# ─────────────────────────────────────────────────────────────────────────────
def plot_probability_heatmaps(agent: DecisionTreeAgent,
                               food_col: int = 5, food_row: int = 3):
    """
    4 heatmaps (une par action) : P(action | position_tête).
    La position de la nourriture est marquée d'une étoile.
    """
    prob_map, best, gap = scan_grid(agent, food_col, food_row)

    fig = plt.figure(figsize=(22, 8), facecolor=BG)
    fig.suptitle(
        f"Probabilités prédites par action — nourriture fixée en ({food_col}, {food_row})\n"
        "Chaque cellule = probabilité que l'agent choisisse cette action depuis cette position",
        fontsize=14, fontweight="bold", color="white", y=1.01
    )

    gs = gridspec.GridSpec(1, N_ACTIONS + 1, figure=fig, wspace=0.30,
                           width_ratios=[1] * N_ACTIONS + [0.08])

    vmin, vmax = 0.0, 1.0

    for ai, aname in enumerate(ACTION_NAMES):
        ax = fig.add_subplot(gs[0, ai])
        ax.set_facecolor(BG)

        im = ax.imshow(prob_map[:, :, ai], cmap=CMAP_PROB,
                       vmin=vmin, vmax=vmax,
                       interpolation="nearest", aspect="auto")

        # Annotation valeurs sur les cellules
        for r in range(GRID_H):
            for c in range(GRID_W):
                v    = prob_map[r, c, ai]
                col  = "white" if v > 0.5 else "#AABBCC"
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        color=col, fontsize=5.5)

        # Étoile = nourriture
        ax.scatter(food_col, food_row, marker="*", s=350,
                   color="#FFD700", zorder=5)

        # Colorbar par action
        cbar_ax = fig.add_subplot(gs[0, N_ACTIONS]) if ai == N_ACTIONS - 1 else None
        if cbar_ax:
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.set_label("P(action)", color=TEXT_COL, fontsize=9)
            cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)

        ax.set_title(aname, color=ACTION_COLORS[ai], fontsize=13,
                     fontweight="bold", pad=9)
        ax.set_xlabel("Colonne", color=TEXT_COL, fontsize=8)
        ax.set_ylabel("Ligne",   color=TEXT_COL, fontsize=8)
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)

    plt.savefig(out("xai_dt_heatmaps.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_dt_heatmaps.png')}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation 2 – Carte de confiance + politique
# ─────────────────────────────────────────────────────────────────────────────
def plot_confidence_map(agent: DecisionTreeAgent,
                        food_col: int = 5, food_row: int = 3):
    """
    Gauche  : heatmap de confiance (prob_max – prob_2nd_max).
              Zones sombres = agent hésitant, claires = agent sûr.
    Droite  : politique apprise (action choisie par cellule).
              Flèches + fond coloré selon l'action + intensité ∝ confiance.
    """
    prob_map, best, gap = scan_grid(agent, food_col, food_row)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
    fig.suptitle(
        "Confiance de l'agent (prob-gap) & Politique apprise\n"
        "Confiance = P_max – P_2nd_max  |  0 = totalement indécis, 1 = certain",
        fontsize=14, fontweight="bold", color="white"
    )

    # ── Heatmap confiance ─────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(BG)
    im = ax1.imshow(gap, cmap=CMAP_GAP, vmin=0, vmax=1,
                    interpolation="nearest", aspect="auto")
    ax1.scatter(food_col, food_row, marker="*", s=400,
                color="#FFD700", zorder=5)

    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("P_max – P_2nd_max", color=TEXT_COL, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    ax1.set_title("Confiance (gap de probabilité)",
                  color="white", fontsize=13, pad=10)
    ax1.set_xlabel("Colonne", color=TEXT_COL, fontsize=9)
    ax1.set_ylabel("Ligne",   color=TEXT_COL, fontsize=9)
    ax1.tick_params(colors="#888888", labelsize=7)
    for spine in ax1.spines.values():
        spine.set_edgecolor(GRID_COL)

    # ── Politique : fond coloré + flèches ────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)

    color_table = {
        0: np.array([0.31, 0.76, 0.97]),   # UP    bleu clair
        1: np.array([0.51, 0.78, 0.52]),   # RIGHT vert
        2: np.array([1.00, 0.72, 0.30]),   # DOWN  orange
        3: np.array([0.94, 0.38, 0.57]),   # LEFT  rose
    }
    policy_rgb = np.zeros((GRID_H, GRID_W, 3))
    for r in range(GRID_H):
        for c in range(GRID_W):
            policy_rgb[r, c] = color_table[best[r, c]]

    gap_norm = (gap - gap.min()) / (gap.max() - gap.min() + 1e-8)
    alpha_map = 0.30 + 0.70 * gap_norm
    for ch in range(3):
        policy_rgb[:, :, ch] *= alpha_map

    ax2.imshow(policy_rgb, interpolation="nearest", aspect="auto")

    arrows = {0: (0, -0.38), 1: (0.38, 0), 2: (0, 0.38), 3: (-0.38, 0)}
    for r in range(GRID_H):
        for c in range(GRID_W):
            dx, dy = arrows[best[r, c]]
            ax2.annotate(
                "", xy=(c + dx, r + dy), xytext=(c, r),
                arrowprops=dict(arrowstyle="->", color="white", lw=0.9)
            )

    ax2.scatter(food_col, food_row, marker="*", s=400,
                color="#FFD700", zorder=5)

    legend_patches = [
        mpatches.Patch(color=tuple(color_table[i]), label=ACTION_NAMES[i])
        for i in range(N_ACTIONS)
    ]
    ax2.legend(handles=legend_patches, loc="upper right", fontsize=8,
               facecolor="#1A1A2E", edgecolor="#444", labelcolor="white")

    ax2.set_title("Politique apprise (action choisie par cellule)",
                  color="white", fontsize=13, pad=10)
    ax2.set_xlabel("Colonne", color=TEXT_COL, fontsize=9)
    ax2.set_ylabel("Ligne",   color=TEXT_COL, fontsize=9)
    ax2.tick_params(colors="#888888", labelsize=7)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID_COL)

    plt.tight_layout()
    plt.savefig(out("xai_dt_confidence.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_dt_confidence.png')}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation 3 – Évolution temporelle des probabilités
# ─────────────────────────────────────────────────────────────────────────────
def plot_temporal_predictions(agent: DecisionTreeAgent,
                               num_episodes: int = 3):
    """
    Lance num_episodes épisodes greedy et enregistre les probabilités prédites
    à chaque step.

    Affiche pour chaque épisode :
      - Courbes des 4 probabilités (une par action)
      - Enveloppe de confiance (prob_max)
      - Marqueurs : ★ = nourriture mangée, ✗ = mort
    """
    game.show           = False
    game.stop_iteration = 2000

    class _ProbCollector:
        def __init__(self, ag):
            self.ag    = ag
            self.steps = []   # (proba [4], action)
            self.events = []  # (step, type)
            self._score = 0
            self._step  = 0

        def tab_state(self, *args):
            return list(args)

        def get_action(self, net, state):
            if len(state) >= 26:
                if   state[22] > 0.5: self.ag.set_direction("UP")
                elif state[23] > 0.5: self.ag.set_direction("RIGHT")
                elif state[24] > 0.5: self.ag.set_direction("DOWN")
                elif state[25] > 0.5: self.ag.set_direction("LEFT")
            proba  = predict_proba(self.ag, np.array(state))[0]
            action = int(np.argmax(proba))
            self.steps.append((proba.copy(), action))
            self._step += 1
            return action

    all_eps = []
    for ep in range(num_episodes):
        col = _ProbCollector(agent)
        agent.direction = "RIGHT"
        score = game.game_loop(
            game.rect_width, game.rect_height, game.display,
            agent, None, 0, col
        )
        all_eps.append({"steps": col.steps, "score": score})
        print(f"[XAI] Épisode {ep+1} terminé — Score : {score} "
              f"({len(col.steps)} steps)")

    fig, axes = plt.subplots(num_episodes, 1,
                             figsize=(20, 5 * num_episodes),
                             facecolor=BG, squeeze=False)
    fig.suptitle(
        "Évolution temporelle des probabilités prédites pendant l'épisode\n"
        "Chaque courbe = P(action | état_courant)  |  "
        "Blanc pointillé = P_max (confiance)  |  "
        "Vert ┊ = nourriture mangée",
        fontsize=14, fontweight="bold", color="white", y=1.01
    )

    for ep_idx, ep_data in enumerate(all_eps):
        ax    = axes[ep_idx, 0]
        ax.set_facecolor(PANEL_BG)
        steps = ep_data["steps"]
        T     = len(steps)
        proba_arr = np.array([p for p, a in steps])   # [T, 4]
        act_arr   = np.array([a for p, a in steps])   # [T]
        t_range   = np.arange(T)

        for ai in range(N_ACTIONS):
            ax.fill_between(t_range, proba_arr[:, ai],
                            alpha=0.07, color=ACTION_COLORS[ai])
            ax.plot(t_range, proba_arr[:, ai],
                    label=ACTION_NAMES[ai], color=ACTION_COLORS[ai],
                    linewidth=1.4, alpha=0.90)

        # P_max (confiance)
        ax.plot(t_range, proba_arr.max(axis=1),
                color="white", linewidth=0.8, linestyle="--",
                alpha=0.45, label="P_max")

        # Marqueurs de changement d'action dominant
        changes = np.where(np.diff(act_arr))[0]
        for ch in changes:
            ax.axvline(x=ch, color="#555566", linewidth=0.6,
                       linestyle=":", alpha=0.5)

        # Score final
        score_final = ep_data["score"]
        ax.set_title(
            f"Épisode {ep_idx + 1}  —  Score final : {score_final}  |  Steps : {T}",
            color="white", fontsize=12, pad=8
        )
        ax.set_xlabel("Step", color=TEXT_COL, fontsize=9)
        ax.set_ylabel("Probabilité prédite", color=TEXT_COL, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.tick_params(colors="#888888", labelsize=8)
        ax.legend(loc="upper left", fontsize=8, facecolor="#0D1117",
                  edgecolor="#444444", labelcolor="white",
                  framealpha=0.8, ncol=5)
        ax.grid(axis="y", color=GRID_COL, linewidth=0.5, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)

        # Action choisie en fond (bandes colorées)
        prev = 0
        for t in range(1, T + 1):
            if t == T or act_arr[t] != act_arr[t - 1]:
                ax.axvspan(prev, t - 0.5, alpha=0.04,
                           color=ACTION_COLORS[act_arr[t - 1]])
                prev = t

    plt.tight_layout()
    plt.savefig(out("xai_dt_temporal.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_dt_temporal.png')}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="XAI – Prédictions et confiance DT Snake"
    )
    parser.add_argument("--heatmap",  action="store_true",
                        help="Heatmaps de probabilité par action")
    parser.add_argument("--gap",      action="store_true",
                        help="Carte de confiance + politique")
    parser.add_argument("--temporal", action="store_true",
                        help="Évolution temporelle des probabilités")
    parser.add_argument("--model",    type=str, default="snake_xgb_model.pkl")
    parser.add_argument("--food-col", type=int, default=5,
                        help="Colonne de la nourriture (défaut : 5)")
    parser.add_argument("--food-row", type=int, default=3,
                        help="Ligne de la nourriture (défaut : 3)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Épisodes pour l'analyse temporelle (défaut : 3)")
    parser.add_argument("--direction", type=str, default="RIGHT",
                        choices=["UP", "RIGHT", "DOWN", "LEFT"],
                        help="Direction du serpent pour les heatmaps (défaut : RIGHT)")
    args = parser.parse_args()

    run_all = not (args.heatmap or args.gap or args.temporal)

    agent = load_agent(args.model)

    if run_all or args.heatmap:
        print("\n[XAI] Génération des heatmaps de probabilité…")
        plot_probability_heatmaps(agent,
                                  food_col=args.food_col,
                                  food_row=args.food_row)

    if run_all or args.gap:
        print("\n[XAI] Génération de la carte de confiance…")
        plot_confidence_map(agent,
                            food_col=args.food_col,
                            food_row=args.food_row)

    if run_all or args.temporal:
        print(f"\n[XAI] Analyse temporelle sur {args.episodes} épisode(s)…")
        plot_temporal_predictions(agent, num_episodes=args.episodes)

    print("\n[XAI] Analyse terminée.")


if __name__ == "__main__":
    main()

# python xai_dt_predictions.py                           # tout
# python xai_dt_predictions.py --heatmap                 # heatmaps
# python xai_dt_predictions.py --gap                     # confiance + politique
# python xai_dt_predictions.py --temporal --episodes 5   # 5 épisodes
# python xai_dt_predictions.py --food-col 8 --food-row 5 # changer la nourriture
# python xai_dt_predictions.py --direction UP            # serpent allant vers le haut
