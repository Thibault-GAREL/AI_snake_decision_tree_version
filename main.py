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

matplotlib.use("Agg")  # rendu sans écran pour les graphiques
import matplotlib.pyplot as plt
import mlflow

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
N_ORACLE_GAMES = 500  # parties jouées par l'heuristique greedy
ORACLE_MAX_STEPS = 2000  # limite de pas par partie

# ── Phase 2 : entraînement initial ────────────────────────────────────────────
MIN_BUFFER_FOR_TRAIN = 3_000  # seuil minimal avant d'entraîner

# ── Phase 3 : DAgger ─────────────────────────────────────────────────────────
N_DAGGER_ROUNDS = 8  # nombre de rounds DAgger
N_GAMES_PER_ROUND = 50  # parties jouées par round par l'agent
DAGGER_BETA_INIT = 0.8  # probabilité initiale de suivre l'oracle (vs agent)
DAGGER_BETA_DECAY = 0.85  # décroissance de beta par round
RETRAIN_EVERY = 2  # ré-entraîner tous les N rounds DAgger

# ── Phase 4 : évaluation ─────────────────────────────────────────────────────
N_EVAL_GAMES = 100  # parties pour évaluation finale

# ── Général ───────────────────────────────────────────────────────────────────
SHOW_GAME = False  # afficher pygame pendant l'entraînement ?
SAVE_PLOTS = True  # sauvegarder les courbes d'apprentissage
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Désactiver l'affichage pygame si on ne veut pas le voir
snake_env.show = SHOW_GAME
snake_env.player = False  # toujours mode IA
snake_env.stop_iteration = ORACLE_MAX_STEPS  # horizon de jeu étendu

# ─────────────────────────────────────────────────────────────────────────────
# Wrapper game_loop avec collecte de données
# ─────────────────────────────────────────────────────────────────────────────


class DataCollectingNeat:
    """
    Remplace l'objet Neat dans snake.py pour collecter (state, action)
    à chaque pas de jeu.
    """

    def __init__(
        self, agent: DecisionTreeAgent, collect_mode: bool = False, beta: float = 0.0
    ):
        """
        agent       : l'agent DecisionTreeAgent
        collect_mode: si True, l'oracle fournit les actions (phase collecte)
        beta        : proba d'utiliser l'oracle au lieu de l'agent (DAgger)
        """
        self.agent = agent
        self.collect_mode = collect_mode
        self.beta = beta
        self.trajectory: list = []  # liste de (state, action, oracle_action)

    def tab_state(self, *args) -> list:
        return list(args)

    def get_action(self, net, state: list) -> int:
        """
        net est ignoré — on utilise self.agent directement.
        """
        # Mise à jour de la direction depuis les features d'état (state[22..25])
        # qui reflètent my_snake.direction côté snake.py.
        # On NE lit pas net.direction car agent.direction n'est jamais mis à jour
        # pendant la partie et resterait toujours "RIGHT".
        if len(state) >= 26:
            if state[22] > 0.5:
                self.agent.set_direction("UP")
            elif state[23] > 0.5:
                self.agent.set_direction("RIGHT")
            elif state[24] > 0.5:
                self.agent.set_direction("DOWN")
            elif state[25] > 0.5:
                self.agent.set_direction("LEFT")
        else:
            self.agent.set_direction(
                net.direction if hasattr(net, "direction") else self.agent.direction
            )

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


def run_game(neat_wrapper: DataCollectingNeat, agent: DecisionTreeAgent) -> int:
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
        agent,  # net  → utilisé pour lire agent.direction
        None,  # genome → non utilisé dans notre cas
        0,  # i → non utilisé
        neat_wrapper,  # Neat → notre wrapper
    )
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Collecte oracle
# ─────────────────────────────────────────────────────────────────────────────


def phase_oracle(agent: DecisionTreeAgent, n_games: int = N_ORACLE_GAMES) -> list:
    """
    Fait jouer l'oracle greedy pendant n_games parties.
    Retourne la liste des scores obtenus.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Collecte oracle ({n_games} parties)")
    print(f"{'='*60}")

    scores = []
    wrapper = DataCollectingNeat(agent, collect_mode=True, beta=1.0)
    t0 = time.time()

    for i in range(n_games):
        agent.direction = "RIGHT"
        score = run_game(wrapper, agent)
        scores.append(score)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  Partie {i+1:4d}/{n_games}  |  "
                f"Score moy : {np.mean(scores[-50:]):.2f}  |  "
                f"Buffer : {len(agent.buffer):6d}  |  "
                f"Temps : {elapsed:.1f}s"
            )

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
        print(
            f"  [WARN] Buffer trop petit ({len(agent.buffer)} < "
            f"{MIN_BUFFER_FOR_TRAIN}). Ajoutez plus de données."
        )
        return 1.0

    error = agent.train(verbose=True)
    agent.stats()
    return error


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — DAgger (dataset aggregation)
# ─────────────────────────────────────────────────────────────────────────────


def phase_dagger(
    agent: DecisionTreeAgent,
    n_rounds: int = N_DAGGER_ROUNDS,
    n_games: int = N_GAMES_PER_ROUND,
    beta_init: float = DAGGER_BETA_INIT,
) -> dict:
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

    beta = beta_init
    history_score = {}
    history_error = {}

    for rnd in range(1, n_rounds + 1):
        print(f"\n  ── Round DAgger {rnd}/{n_rounds}  (beta={beta:.3f}) ──")

        wrapper = DataCollectingNeat(agent, collect_mode=False, beta=beta)
        scores = []

        for i in range(n_games):
            agent.direction = "RIGHT"
            score = run_game(wrapper, agent)
            scores.append(score)

        mean_score = float(np.mean(scores))
        max_score = int(np.max(scores))
        print(
            f"    Score moyen : {mean_score:.2f}  |  "
            f"Max : {max_score}  |  Buffer : {len(agent.buffer)}"
        )

        history_score[rnd] = mean_score

        # Ré-entraînement périodique
        if rnd % RETRAIN_EVERY == 0:
            print(f"    → Ré-entraînement...")
            error = agent.train(verbose=False)
            history_error[rnd] = error
            print(
                f"    → Erreur CV-3 : {error:.4f}  "
                f"(Accuracy ≈ {(1-error)*100:.1f}%)"
            )

        # Décroissance de beta
        beta *= DAGGER_BETA_DECAY
        beta = max(beta, 0.05)  # garder 5% d'oracle minimum

    return {"scores": history_score, "errors": history_error}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Évaluation
# ─────────────────────────────────────────────────────────────────────────────


def phase_eval(agent: DecisionTreeAgent, n_games: int = N_EVAL_GAMES) -> dict:
    """
    Évalue l'agent final sur n_games parties (sans collecte de données).
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 4 — Évaluation finale ({n_games} parties)")
    print(f"{'='*60}")

    # Wrapper en mode "agent pur" (beta=0, collect_mode=False)
    wrapper = DataCollectingNeat(agent, collect_mode=False, beta=0.0)
    scores = []
    t0 = time.time()

    for i in range(n_games):
        agent.direction = "RIGHT"
        score = run_game(wrapper, agent)
        scores.append(score)

        if (i + 1) % 25 == 0:
            print(
                f"  Partie {i+1:4d}/{n_games}  |  "
                f"Score moy : {np.mean(scores):.2f}  |  "
                f"Max : {max(scores)}"
            )

    results = {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "max": int(np.max(scores)),
        "median": float(np.median(scores)),
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


def plot_results(
    oracle_scores: list,
    dagger_history: dict,
    eval_results: dict,
    save_path: str = "snake_training_curves.png",
):
    """Génère et sauvegarde les courbes d'apprentissage."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Snake — Entraînement Arbre de Décision (XGBoost)",
        fontsize=14,
        fontweight="bold",
    )

    # ── 1. Scores oracle ──────────────────────────────────────────────────
    ax = axes[0]
    window = 20
    if len(oracle_scores) >= window:
        smoothed = np.convolve(oracle_scores, np.ones(window) / window, mode="valid")
        ax.plot(oracle_scores, alpha=0.3, color="steelblue", label="Brut")
        ax.plot(
            range(window - 1, len(oracle_scores)),
            smoothed,
            color="steelblue",
            linewidth=2,
            label=f"Moyenne {window}",
        )
    else:
        ax.plot(oracle_scores, color="steelblue", linewidth=2, label="Brut")
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
    ax.hist(
        eval_results["scores"], bins=15, color="seagreen", edgecolor="white", alpha=0.85
    )
    ax.axvline(
        eval_results["mean"],
        color="red",
        linestyle="--",
        label=f"Moyenne {eval_results['mean']:.1f}",
    )
    ax.axvline(
        eval_results["median"],
        color="orange",
        linestyle=":",
        label=f"Médiane {eval_results['median']:.1f}",
    )
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
    """Lance le pipeline complet d'entraînement avec MLflow tracking."""
    print("\n" + "█" * 60)
    print("  SNAKE — Agent Arbre de Décision (XGBoost + CUDA)")
    print("█" * 60)

    # Création de l'agent
    agent = DecisionTreeAgent(use_cuda=True)

    # Chargement si un modèle existe déjà
    agent.load()

    # ── MLflow setup ──────────────────────────────────────────────────────
    mlflow.set_experiment("snake-decision-tree")

    with mlflow.start_run(run_name="xgboost-dagger-full-pipeline"):

        # Log tous les hyperparamètres
        mlflow.log_params({
            # Phase 1
            "N_ORACLE_GAMES": N_ORACLE_GAMES,
            "ORACLE_MAX_STEPS": ORACLE_MAX_STEPS,
            # Phase 2
            "MIN_BUFFER_FOR_TRAIN": MIN_BUFFER_FOR_TRAIN,
            # Phase 3
            "N_DAGGER_ROUNDS": N_DAGGER_ROUNDS,
            "N_GAMES_PER_ROUND": N_GAMES_PER_ROUND,
            "DAGGER_BETA_INIT": DAGGER_BETA_INIT,
            "DAGGER_BETA_DECAY": DAGGER_BETA_DECAY,
            "RETRAIN_EVERY": RETRAIN_EVERY,
            # Phase 4
            "N_EVAL_GAMES": N_EVAL_GAMES,
            # Général
            "RANDOM_SEED": RANDOM_SEED,
            # XGBoost
            "xgb_n_estimators": 400,
            "xgb_max_depth": 8,
            "xgb_learning_rate": 0.05,
            "xgb_subsample": 0.85,
            "xgb_colsample_bytree": 0.85,
            "xgb_min_child_weight": 3,
            "xgb_gamma": 0.1,
            "xgb_reg_alpha": 0.05,
            "xgb_reg_lambda": 1.5,
            "xgb_tree_method": "hist",
            # Environnement
            "backend": "XGBoost CUDA" if agent.use_cuda else "Sklearn CPU",
        })

        # Log versions des librairies
        import xgboost as xgb_mod
        import sklearn
        mlflow.log_params({
            "python_version": sys.version.split()[0],
            "xgboost_version": xgb_mod.__version__,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "mlflow_version": mlflow.__version__,
        })

        total_start = time.time()

        # ── Phase 1 : collecte oracle ─────────────────────────────────────
        t_phase1 = time.time()
        if len(agent.buffer) < MIN_BUFFER_FOR_TRAIN:
            oracle_scores = phase_oracle(agent, n_games=N_ORACLE_GAMES)
        else:
            print(
                f"\n[Skip] Buffer déjà riche ({len(agent.buffer)} exemples). "
                "Phase oracle ignorée."
            )
            oracle_scores = [0]
        phase1_time = time.time() - t_phase1

        mlflow.log_metrics({
            "phase1_oracle_mean_score": float(np.mean(oracle_scores)),
            "phase1_oracle_std_score": float(np.std(oracle_scores)),
            "phase1_oracle_max_score": float(np.max(oracle_scores)),
            "phase1_buffer_size": len(agent.buffer),
            "phase1_time_s": phase1_time,
        })

        # ── Phase 2 : entraînement initial ────────────────────────────────
        t_phase2 = time.time()
        if not agent.trained:
            phase_train(agent)
            agent.save()  # sauvegarde intermédiaire après Phase 2
        else:
            print("\n[Skip] Modèle déjà entraîné. Passe directement au DAgger.")
        phase2_time = time.time() - t_phase2

        if agent.train_history:
            last = agent.train_history[-1]
            mlflow.log_metrics({
                "phase2_train_error_rate": last["error_rate"],
                "phase2_train_accuracy": 1 - last["error_rate"],
                "phase2_train_samples": last["n_samples"],
            })
        mlflow.log_metric("phase2_time_s", phase2_time)

        # ── Phase 3 : DAgger ──────────────────────────────────────────────
        t_phase3 = time.time()
        dagger_history = phase_dagger(
            agent,
            n_rounds=N_DAGGER_ROUNDS,
            n_games=N_GAMES_PER_ROUND,
            beta_init=DAGGER_BETA_INIT,
        )
        phase3_time = time.time() - t_phase3

        # Log DAgger metrics par round
        for rnd, score in dagger_history["scores"].items():
            mlflow.log_metric("dagger_mean_score", score, step=rnd)
        for rnd, error in dagger_history["errors"].items():
            mlflow.log_metric("dagger_cv3_error", error, step=rnd)
            mlflow.log_metric("dagger_cv3_accuracy", 1 - error, step=rnd)

        mlflow.log_metrics({
            "phase3_buffer_size": len(agent.buffer),
            "phase3_time_s": phase3_time,
        })

        # ── Phase 4 : évaluation ──────────────────────────────────────────
        t_phase4 = time.time()
        eval_results = phase_eval(agent, n_games=N_EVAL_GAMES)
        phase4_time = time.time() - t_phase4

        mlflow.log_metrics({
            "eval_mean_score": eval_results["mean"],
            "eval_std_score": eval_results["std"],
            "eval_max_score": eval_results["max"],
            "eval_median_score": eval_results["median"],
            "phase4_time_s": phase4_time,
        })

        # ── Sauvegarde ────────────────────────────────────────────────────
        agent.save()

        # ── Visualisation ─────────────────────────────────────────────────
        if SAVE_PLOTS:
            plot_results(oracle_scores, dagger_history, eval_results)

        total_time = time.time() - total_start

        mlflow.log_metric("total_time_s", total_time)

        # Log artifacts
        if os.path.exists("snake_xgb_model.pkl"):
            mlflow.log_artifact("snake_xgb_model.pkl")
        if os.path.exists("snake_training_curves.png"):
            mlflow.log_artifact("snake_training_curves.png")

        print(f"\n[Done] Temps total d'entraînement : {total_time:.1f}s")
        print(f"[Done] Score final moyen : {eval_results['mean']:.2f}")
        print(f"[MLflow] Run ID : {mlflow.active_run().info.run_id}")

    return agent, eval_results


def demo_mode(n_games: int = 5):
    """Charge un modèle existant et joue quelques parties en mode visible."""
    print("\n[Demo] Chargement du modèle sauvegardé...")
    agent = DecisionTreeAgent(use_cuda=True)
    agent.load()

    if not agent.trained:
        print(
            "[Demo] Aucun modèle entraîné trouvé. "
            "Lancez d'abord le mode entraînement."
        )
        return

    # Activer l'affichage pygame pour la démo
    snake_env.show = True
    snake_env.player = False
    snake_env.vitesse = 8

    if not hasattr(snake_env, "clock") or snake_env.display is None:
        import pygame

        pygame.init()
        snake_env.display = pygame.display.set_mode(
            (snake_env.width, int(snake_env.height))
        )
        pygame.display.set_caption("Snake — Agent Arbre de Décision")
        snake_env.clock = pygame.time.Clock()
        snake_env.fonttype = pygame.font.SysFont(None, 30)

    wrapper = DataCollectingNeat(agent, collect_mode=False, beta=0.0)
    scores = []

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
