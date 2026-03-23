"""
main.py
=======
Point d'entrée principal.

Phases d'exécution :
  1. PHASE_ORACLE  : collecte de données via l'agent heuristique greedy
                     (N_ORACLE_GAMES parties).
  2. PHASE_TRAIN   : entraînement initial du modèle XGBoost/GradientBoosting.
  3. PHASE_DAGGER  : boucles DAgger-light — l'agent joue, les bons épisodes
                     sont ajoutés au buffer, puis on ré-entraîne.
  4. PHASE_EVAL    : évaluation finale sur N_EVAL_GAMES parties.

Le fichier s'interface directement avec snake.py via game_loop().
"""

import sys
import os
import time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")   # rendu sans écran pour les graphiques
import matplotlib.pyplot as plt

# ── Import du moteur snake ────────────────────────────────────────────────────
# On réutilise game_loop et les constantes définies dans snake.py
import snake as snake_env

# ── Import de notre agent ────────────────────────────────────────────────────
from arbre_de_decision import (
    DecisionTreeAgent,
    NeatCompatibleWrapper,
    DIRECTION_MAP,
    OPPOSITE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparamètres d'entraînement
# ─────────────────────────────────────────────────────────────────────────────

# ── Phase 1 : collecte oracle ─────────────────────────────────────────────────
N_ORACLE_GAMES      = 300     # parties jouées par l'heuristique greedy
ORACLE_MAX_STEPS    = 500     # limite de pas par partie (= stop_iteration)

# ── Phase 2 : entraînement initial ────────────────────────────────────────────
MIN_BUFFER_FOR_TRAIN = 3_000  # seuil minimal avant d'entraîner

# ── Phase 3 : DAgger ─────────────────────────────────────────────────────────
N_DAGGER_ROUNDS      = 10     # nombre de rounds DAgger
N_GAMES_PER_ROUND    = 50     # parties jouées par round par l'agent
DAGGER_BETA_INIT     = 0.8    # probabilité initiale de suivre l'oracle (vs agent)
DAGGER_BETA_DECAY    = 0.85   # décroissance de beta par round
RETRAIN_EVERY        = 2      # ré-entraîner tous les N rounds DAgger

# ── Phase 4 : évaluation ─────────────────────────────────────────────────────
N_EVAL_GAMES         = 100    # parties pour évaluation finale

# ── Général ───────────────────────────────────────────────────────────────────
SHOW_GAME            = False   # afficher pygame pendant l'entraînement ?
SAVE_PLOTS           = True    # sauvegarder les courbes d'apprentissage
RANDOM_SEED          = 42

# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Désactiver l'affichage pygame si on ne veut pas le voir
snake_env.show   = SHOW_GAME
snake_env.player = False          # toujours mode IA

# ─────────────────────────────────────────────────────────────────────────────
# Wrapper game_loop avec collecte de données
# ─────────────────────────────────────────────────────────────────────────────

class DataCollectingNeat:
    """
    Remplace l'objet Neat dans snake.py pour collecter (state, action)
    à chaque pas de jeu.
    """
    def __init__(self, agent: DecisionTreeAgent,
                 collect_mode: bool = False,
                 beta: float = 0.0):
        """
        agent       : l'agent DecisionTreeAgent
        collect_mode: si True, l'oracle fournit les actions (phase collecte)
        beta        : proba d'utiliser l'oracle au lieu de l'agent (DAgger)
        """
        self.agent        = agent
        self.collect_mode = collect_mode
        self.beta         = beta
        self.trajectory: list = []   # liste de (state, action, oracle_action)

    def tab_state(self, *args) -> list:
        return list(args)

    def get_action(self, net, state: list) -> int:
        """
        net est ignoré — on utilise self.agent directement.
        """
        # Mise à jour de la direction dans l'agent
        # (snake_env gère la direction en interne, on la lit depuis net qui
        #  est notre agent)
        self.agent.set_direction(net.direction if hasattr(net, "direction")
                                 else self.agent.direction)

        # Mode collecte pure : oracle fournit toujours l'action
        if self.collect_mode:
            action = self.agent.record_oracle_step(state)
            self.trajectory.append((state, action, action))
            return action

        # Mode DAgger : mélange oracle / agent selon beta
        oracle_action = self.agent.oracle.get_action(state)
        if random.random() < self.beta:
            action = oracle_action
        else:
            action = self.agent.get_action(state)

        # On enregistre toujours l'action oracle comme label supervisé
        self.agent.buffer.push(state, oracle_action)
        self.trajectory.append((state, action, oracle_action))
        return action


def run_game(neat_wrapper: DataCollectingNeat,
             agent: DecisionTreeAgent) -> int:
    """
    Lance une partie de snake et retourne le score.
    Utilise le wrapper DataCollectingNeat comme 'Neat'.
    """
    # game_loop attend : (rect_width, rect_height, display, net, genome, i, Neat)
    # net   = notre agent (pour lire la direction)
    # Neat  = notre wrapper (tab_state + get_action)
    score = snake_env.game_loop(
        snake_env.rect_width,
        snake_env.rect_height,
        snake_env.display,
        agent,          # net  → utilisé pour lire agent.direction
        None,           # genome → non utilisé dans notre cas
        0,              # i → non utilisé
        neat_wrapper,   # Neat → notre wrapper
    )
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Collecte oracle
# ─────────────────────────────────────────────────────────────────────────────

def phase_oracle(agent: DecisionTreeAgent,
                 n_games: int = N_ORACLE_GAMES) -> list:
    """
    Fait jouer l'oracle greedy pendant n_games parties.
    Retourne la liste des scores obtenus.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Collecte oracle ({n_games} parties)")
    print(f"{'='*60}")

    scores  = []
    wrapper = DataCollectingNeat(agent, collect_mode=True, beta=1.0)
    t0      = time.time()

    for i in range(n_games):
        agent.direction = "RIGHT"
        score = run_game(wrapper, agent)
        scores.append(score)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Partie {i+1:4d}/{n_games}  |  "
                  f"Score moy : {np.mean(scores[-50:]):.2f}  |  "
                  f"Buffer : {len(agent.buffer):6d}  |  "
                  f"Temps : {elapsed:.1f}s")

    print(f"\n  → Score moyen oracle : {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"  → Buffer total : {len(agent.buffer)} exemples")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Entraînement initial
# ─────────────────────────────────────────────────────────────────────────────

def phase_train(agent: DecisionTreeAgent) -> float:
    """Entraîne le modèle sur le buffer courant."""
    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Entraînement initial")
    print(f"{'='*60}")

    if len(agent.buffer) < MIN_BUFFER_FOR_TRAIN:
        print(f"  [WARN] Buffer trop petit ({len(agent.buffer)} < "
              f"{MIN_BUFFER_FOR_TRAIN}). Ajoutez plus de données.")
        return 1.0

    error = agent.train(verbose=True)
    agent.stats()
    return error


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — DAgger (dataset aggregation)
# ─────────────────────────────────────────────────────────────────────────────

def phase_dagger(agent: DecisionTreeAgent,
                 n_rounds: int   = N_DAGGER_ROUNDS,
                 n_games:  int   = N_GAMES_PER_ROUND,
                 beta_init: float = DAGGER_BETA_INIT) -> dict:
    """
    DAgger-light :
      - L'agent joue des parties avec un mélange oracle/agent (ratio beta)
      - Les labels supervisés viennent toujours de l'oracle
      - On ré-entraîne périodiquement
    Retourne l'historique des scores par round.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 3 — DAgger ({n_rounds} rounds × {n_games} parties)")
    print(f"{'='*60}")

    beta          = beta_init
    history_score = {}
    history_error = {}

    for rnd in range(1, n_rounds + 1):
        print(f"\n  ── Round DAgger {rnd}/{n_rounds}  (beta={beta:.3f}) ──")

        wrapper = DataCollectingNeat(agent, collect_mode=False, beta=beta)
        scores  = []

        for i in range(n_games):
            agent.direction = "RIGHT"
            score = run_game(wrapper, agent)
            scores.append(score)

        mean_score = float(np.mean(scores))
        max_score  = int(np.max(scores))
        print(f"    Score moyen : {mean_score:.2f}  |  "
              f"Max : {max_score}  |  Buffer : {len(agent.buffer)}")

        history_score[rnd] = mean_score

        # Ré-entraînement périodique
        if rnd % RETRAIN_EVERY == 0:
            print(f"    → Ré-entraînement...")
            error = agent.train(verbose=False)
            history_error[rnd] = error
            print(f"    → Erreur CV-3 : {error:.4f}  "
                  f"(Accuracy ≈ {(1-error)*100:.1f}%)")

        # Décroissance de beta
        beta *= DAGGER_BETA_DECAY
        beta  = max(beta, 0.05)   # garder 5% d'oracle minimum

    return {"scores": history_score, "errors": history_error}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Évaluation
# ─────────────────────────────────────────────────────────────────────────────

def phase_eval(agent: DecisionTreeAgent,
               n_games: int = N_EVAL_GAMES) -> dict:
    """
    Évalue l'agent final sur n_games parties (sans collecte de données).
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 4 — Évaluation finale ({n_games} parties)")
    print(f"{'='*60}")

    # Wrapper en mode "agent pur" (beta=0, collect_mode=False)
    wrapper = DataCollectingNeat(agent, collect_mode=False, beta=0.0)
    scores  = []
    t0      = time.time()

    for i in range(n_games):
        agent.direction = "RIGHT"
        score = run_game(wrapper, agent)
        scores.append(score)

        if (i + 1) % 25 == 0:
            print(f"  Partie {i+1:4d}/{n_games}  |  "
                  f"Score moy : {np.mean(scores):.2f}  |  "
                  f"Max : {max(scores)}")

    results = {
        "scores":  scores,
        "mean":    float(np.mean(scores)),
        "std":     float(np.std(scores)),
        "max":     int(np.max(scores)),
        "median":  float(np.median(scores)),
        "elapsed": time.time() - t0,
    }

    print(f"\n  ── Résultats finaux ──────────────────────────────────")
    print(f"  Score moyen  : {results['mean']:.2f} ± {results['std']:.2f}")
    print(f"  Score médian : {results['median']:.1f}")
    print(f"  Score max    : {results['max']}")
    print(f"  Temps total  : {results['elapsed']:.1f}s")
    print(f"  ─────────────────────────────────────────────────────")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation des courbes
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(oracle_scores: list,
                 dagger_history: dict,
                 eval_results: dict,
                 save_path: str = "snake_training_curves.png"):
    """Génère et sauvegarde les courbes d'apprentissage."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Snake — Entraînement Arbre de Décision (XGBoost)",
                 fontsize=14, fontweight="bold")

    # ── 1. Scores oracle ──────────────────────────────────────────────────
    ax = axes[0]
    window = 20
    smoothed = np.convolve(oracle_scores,
                           np.ones(window) / window, mode="valid")
    ax.plot(oracle_scores, alpha=0.3, color="steelblue", label="Brut")
    ax.plot(range(window - 1, len(oracle_scores)),
            smoothed, color="steelblue", linewidth=2, label=f"Moyenne {window}")
    ax.set_title("Phase 1 — Scores Oracle")
    ax.set_xlabel("Partie")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 2. Progression DAgger ─────────────────────────────────────────────
    ax = axes[1]
    rounds = sorted(dagger_history["scores"].keys())
    scores = [dagger_history["scores"][r] for r in rounds]
    ax.plot(rounds, scores, "o-", color="darkorange", linewidth=2, markersize=6)
    ax.set_title("Phase 3 — Progression DAgger")
    ax.set_xlabel("Round DAgger")
    ax.set_ylabel("Score moyen / round")
    ax.grid(alpha=0.3)

    # ── 3. Distribution scores finaux ────────────────────────────────────
    ax = axes[2]
    ax.hist(eval_results["scores"], bins=15, color="seagreen",
            edgecolor="white", alpha=0.85)
    ax.axvline(eval_results["mean"],   color="red",    linestyle="--",
               label=f"Moyenne {eval_results['mean']:.1f}")
    ax.axvline(eval_results["median"], color="orange", linestyle=":",
               label=f"Médiane {eval_results['median']:.1f}")
    ax.set_title("Phase 4 — Distribution Scores Finaux")
    ax.set_xlabel("Score")
    ax.set_ylabel("Fréquence")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Courbes sauvegardées dans {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────────────────────────────────────

def train_pipeline():
    """Lance le pipeline complet d'entraînement."""
    print("\n" + "█" * 60)
    print("  SNAKE — Agent Arbre de Décision (XGBoost + CUDA)")
    print("█" * 60)

    # Création de l'agent
    agent = DecisionTreeAgent(use_cuda=True)

    # Chargement si un modèle existe déjà
    agent.load()

    total_start = time.time()

    # ── Phase 1 : collecte oracle ─────────────────────────────────────────
    if len(agent.buffer) < MIN_BUFFER_FOR_TRAIN:
        oracle_scores = phase_oracle(agent, n_games=N_ORACLE_GAMES)
    else:
        print(f"\n[Skip] Buffer déjà riche ({len(agent.buffer)} exemples). "
              "Phase oracle ignorée.")
        oracle_scores = [0]

    # ── Phase 2 : entraînement initial ───────────────────────────────────
    if not agent.trained:
        phase_train(agent)
    else:
        print("\n[Skip] Modèle déjà entraîné. Passe directement au DAgger.")

    # ── Phase 3 : DAgger ─────────────────────────────────────────────────
    dagger_history = phase_dagger(
        agent,
        n_rounds  = N_DAGGER_ROUNDS,
        n_games   = N_GAMES_PER_ROUND,
        beta_init = DAGGER_BETA_INIT,
    )

    # ── Phase 4 : évaluation ─────────────────────────────────────────────
    eval_results = phase_eval(agent, n_games=N_EVAL_GAMES)

    # ── Sauvegarde ───────────────────────────────────────────────────────
    agent.save()

    # ── Visualisation ────────────────────────────────────────────────────
    if SAVE_PLOTS:
        plot_results(oracle_scores, dagger_history, eval_results)

    total_time = time.time() - total_start
    print(f"\n[Done] Temps total d'entraînement : {total_time:.1f}s")
    print(f"[Done] Score final moyen : {eval_results['mean']:.2f}")
    return agent, eval_results


def demo_mode(n_games: int = 5):
    """Charge un modèle existant et joue quelques parties en mode visible."""
    print("\n[Demo] Chargement du modèle sauvegardé...")
    agent = DecisionTreeAgent(use_cuda=True)
    agent.load()

    if not agent.trained:
        print("[Demo] Aucun modèle entraîné trouvé. "
              "Lancez d'abord le mode entraînement.")
        return

    # Activer l'affichage pygame pour la démo
    snake_env.show   = True
    snake_env.player = False
    snake_env.vitesse = 8

    if not hasattr(snake_env, 'clock') or snake_env.display is None:
        import pygame
        pygame.init()
        snake_env.display = pygame.display.set_mode(
            (snake_env.width, int(snake_env.height)))
        pygame.display.set_caption("Snake — Agent Arbre de Décision")
        snake_env.clock    = pygame.time.Clock()
        snake_env.fonttype = pygame.font.SysFont(None, 30)

    wrapper = DataCollectingNeat(agent, collect_mode=False, beta=0.0)
    scores  = []

    print(f"[Demo] Démarrage de {n_games} parties...")
    for i in range(n_games):
        agent.direction = "RIGHT"
        score = run_game(wrapper, agent)
        scores.append(score)
        print(f"  Partie {i+1}/{n_games}  →  Score : {score}")

    print(f"\n[Demo] Score moyen : {np.mean(scores):.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage :
      python main.py            → entraîne le modèle
      python main.py demo       → charge le modèle et joue en visuel
      python main.py demo 10    → idem, 10 parties
    """
    args = sys.argv[1:]

    if args and args[0] == "demo":
        n = int(args[1]) if len(args) > 1 else 5
        demo_mode(n_games=n)
    else:
        agent, results = train_pipeline()
