"""
arbre_de_decision.py
====================
Agent Snake basé sur des arbres de décision boostés (XGBoost + GPU/CUDA).

Architecture d'apprentissage :
  1. Phase de collecte  : on fait jouer un agent heuristique "greedy" pour
     générer un jeu de données (state → action).
  2. Phase d'entraînement : XGBoost multi-classes (4 actions) entraîné avec
     tree_method='hist' + device='cuda' si GPU disponible.
  3. Phase de self-play   : l'agent joue, les trajectoires récompensées sont
     ajoutées au dataset, puis on ré-entraîne (DAgger-light).

Features (16 valeurs flottantes) identiques à celles de snake.py :
  [0]  distance_bord_north          [8]  distance_food_north
  [1]  distance_bord_north_est      [9]  distance_food_north_est
  [2]  distance_bord_est            [10] distance_food_est
  [3]  distance_bord_south_est      [11] distance_food_south_est
  [4]  distance_bord_south          [12] distance_food_south
  [5]  distance_bord_south_west     [13] distance_food_south_west
  [6]  distance_bord_west           [14] distance_food_west
  [7]  distance_bord_north_west     [15] distance_food_north_west

Actions : 0=UP  1=RIGHT  2=DOWN  3=LEFT
"""

import os
import random
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# ── XGBoost (GPU via device='cuda') ──────────────────────────────────────────
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] xgboost non installé. Utilisation de sklearn DecisionTree "
          "comme fallback (pas de GPU).")

# ── Sklearn fallback + utilitaires ───────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
N_FEATURES      = 26   # 16 dist + 2 food_delta + 4 danger_bin + 4 dir_one_hot
N_ACTIONS       = 4
ACTION_NAMES    = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
DIRECTION_MAP   = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}
OPPOSITE        = {0: 2, 1: 3, 2: 0, 3: 1}   # action → action opposée

# ── Hyperparamètres XGBoost (optimisés pour Snake) ───────────────────────────
XGBOOST_PARAMS = {
    "objective":          "multi:softmax",
    "num_class":          N_ACTIONS,
    "tree_method":        "hist",       # 'hist' = approx histogram, compatible GPU
    "device":             "cuda",       # GPU CUDA ; tombera en CPU si absent
    "n_estimators":       400,          # nombre de boosting rounds
    "max_depth":          8,            # profondeur max de chaque arbre
    "learning_rate":      0.05,         # eta  — faible pour meilleure généralisation
    "subsample":          0.85,         # fraction d'échantillons par arbre
    "colsample_bytree":   0.85,         # fraction de features par arbre
    "min_child_weight":   3,            # régularisation sur les feuilles
    "gamma":              0.1,          # seuil de gain min pour split
    "reg_alpha":          0.05,         # L1 régularisation
    "reg_lambda":         1.5,          # L2 régularisation
    "scale_pos_weight":   1,
    "seed":               42,
    "verbosity":          0,
    "use_label_encoder":  False,
    "eval_metric":        "merror",     # multi-class error
}

# ── Hyperparamètres fallback GradientBoosting sklearn ────────────────────────
SKLEARN_PARAMS = {
    "n_estimators":   300,
    "max_depth":      7,
    "learning_rate":  0.05,
    "subsample":      0.85,
    "min_samples_leaf": 4,
    "random_state":   42,
}


# ─────────────────────────────────────────────────────────────────────────────
# Détection GPU
# ─────────────────────────────────────────────────────────────────────────────
def detect_cuda() -> bool:
    """Retourne True si XGBoost peut utiliser CUDA."""
    if not XGBOOST_AVAILABLE:
        return False
    try:
        # Test rapide : entraîner 1 arbre sur données minuscules
        dummy_X = np.random.rand(10, 4).astype(np.float32)
        dummy_y = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        dtrain = xgb.DMatrix(dummy_X, label=dummy_y)
        params = {"tree_method": "hist", "device": "cuda",
                  "objective": "multi:softmax", "num_class": 4,
                  "verbosity": 0}
        xgb.train(params, dtrain, num_boost_round=1)
        print("[INFO] CUDA détecté — XGBoost utilisera le GPU.")
        return True
    except Exception:
        print("[INFO] CUDA non disponible — XGBoost utilisera le CPU.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Buffer de replay
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """Stocke les paires (state, action) pour l'entraînement supervisé."""

    def __init__(self, max_size: int = 200_000):
        self.max_size  = max_size
        self.states:   List[List[float]] = []
        self.actions:  List[int]         = []

    def push(self, state: List[float], action: int):
        self.states.append(state)
        self.actions.append(action)
        # FIFO si dépassement
        if len(self.states) > self.max_size:
            self.states.pop(0)
            self.actions.pop(0)

    def push_batch(self, states: List[List[float]], actions: List[int]):
        for s, a in zip(states, actions):
            self.push(s, a)

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        idx = random.sample(range(len(self.states)), min(n, len(self.states)))
        X = np.array([self.states[i]  for i in idx], dtype=np.float32)
        y = np.array([self.actions[i] for i in idx], dtype=np.int32)
        return X, y

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        return (np.array(self.states,  dtype=np.float32),
                np.array(self.actions, dtype=np.int32))

    def __len__(self) -> int:
        return len(self.states)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"states": self.states, "actions": self.actions}, f)
        print(f"[Buffer] Sauvegardé dans {path}  ({len(self)} exemples)")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.states  = data["states"]
        self.actions = data["actions"]
        print(f"[Buffer] Chargé depuis {path}  ({len(self)} exemples)")


# ─────────────────────────────────────────────────────────────────────────────
# Agent heuristique (Oracle greedy) — génère les données initiales
# ─────────────────────────────────────────────────────────────────────────────
class GreedyOracle:
    """
    Agent heuristique simple : va vers la nourriture en évitant
    les obstacles immédiats. Sert à amorcer le dataset.
    """

    def __init__(self):
        self.direction = "RIGHT"   # suivi de la direction courante

    def set_direction(self, direction: str):
        self.direction = direction

    def get_action(self, state: List[float]) -> int:
        """
        state[0..7]  : distances obstacle (N NE E SE S SW W NW)
        state[8..15] : distances nourriture alignée (N NE E SE S SW W NW)
        state[16]    : food_delta_x normalisé  (+= nourriture à droite)
        state[17]    : food_delta_y normalisé  (+= nourriture en bas)
        state[18..21]: danger binaire immédiat (N E S W)
        state[22..25]: direction courante one-hot (UP RIGHT DOWN LEFT)
        """
        current_dir = DIRECTION_MAP.get(self.direction, 1)

        if len(state) >= 26:
            # ── Oracle enrichi (features v2) ──────────────────────────────
            food_dx     = state[16]   # >0 → nourriture à droite
            food_dy     = state[17]   # >0 → nourriture en bas
            d_north     = state[18]
            d_east      = state[19]
            d_south     = state[20]
            d_west      = state[21]
            safe = {0: not d_north, 1: not d_east, 2: not d_south, 3: not d_west}

            # Candidats vers la nourriture (on exclut direction opposée)
            candidates = []
            if food_dy < -0.01 and safe[0]:
                candidates.append((0, abs(food_dy)))
            if food_dx > 0.01  and safe[1]:
                candidates.append((1, abs(food_dx)))
            if food_dy > 0.01  and safe[2]:
                candidates.append((2, abs(food_dy)))
            if food_dx < -0.01 and safe[3]:
                candidates.append((3, abs(food_dx)))

            # Filtrer la direction opposée
            candidates = [(a, d) for a, d in candidates
                          if a != OPPOSITE[current_dir]]

            if candidates:
                return max(candidates, key=lambda x: x[1])[0]

            # Fallback : direction sûre non-opposée (préférer tout droit)
            safe_actions = [a for a in range(4)
                            if safe[a] and a != OPPOSITE[current_dir]]
            if current_dir in safe_actions:
                return current_dir
            if safe_actions:
                # Préférer la direction la plus dégagée (distance obstacle max)
                dist_map = {0: state[0], 1: state[2], 2: state[4], 3: state[6]}
                return max(safe_actions, key=lambda a: dist_map[a])

        else:
            # ── Oracle legacy (features v1, 16 features) ──────────────────
            SAFE_DIST = 50
            danger_up    = state[0]
            danger_right = state[2]
            danger_down  = state[4]
            danger_left  = state[6]
            food_up    = state[8]
            food_right = state[10]
            food_down  = state[12]
            food_left  = state[14]

            candidates = []
            if food_up    > 0 and danger_up    > SAFE_DIST:
                candidates.append((0, food_up))
            if food_right > 0 and danger_right > SAFE_DIST:
                candidates.append((1, food_right))
            if food_down  > 0 and danger_down  > SAFE_DIST:
                candidates.append((2, food_down))
            if food_left  > 0 and danger_left  > SAFE_DIST:
                candidates.append((3, food_left))

            if candidates:
                best = max(candidates, key=lambda x: x[1])
                if best[0] != OPPOSITE[current_dir]:
                    return best[0]

            safe_actions = []
            for a, dist in [(0, danger_up), (1, danger_right),
                            (2, danger_down), (3, danger_left)]:
                if a != OPPOSITE[current_dir] and dist > SAFE_DIST:
                    safe_actions.append((a, dist))

            if safe_actions:
                straight = [x for x in safe_actions if x[0] == current_dir]
                if straight:
                    return straight[0][0]
                return max(safe_actions, key=lambda x: x[1])[0]

        # Dernier recours : action aléatoire non-suicide
        non_opposite = [a for a in range(4) if a != OPPOSITE[current_dir]]
        return random.choice(non_opposite)


# ─────────────────────────────────────────────────────────────────────────────
# Agent principal — Arbre de décision boosté
# ─────────────────────────────────────────────────────────────────────────────
class DecisionTreeAgent:
    """
    Agent Snake entraîné avec XGBoost (GPU) ou GradientBoosting (CPU fallback).

    Cycle d'entraînement :
      1. collect_oracle_data()  — jouer N parties avec l'oracle greedy
      2. train()                — entraîner le modèle sur le buffer
      3. self_play_and_retrain()— DAgger-light : jouer + ré-entraîner
    """

    MODEL_PATH  = "snake_xgb_model.pkl"
    BUFFER_PATH = "snake_replay_buffer.pkl"

    def __init__(self, use_cuda: bool = True):
        self.use_cuda  = use_cuda and detect_cuda()
        self.model     = None
        self.scaler    = StandardScaler()
        self.buffer    = ReplayBuffer(max_size=300_000)
        self.oracle    = GreedyOracle()
        self.trained   = False
        self.direction = "RIGHT"   # état courant de la direction du serpent

        # Statistiques d'entraînement
        self.train_history: List[dict] = []

    # ── Inférence ─────────────────────────────────────────────────────────
    def get_action(self, state: List[float]) -> int:
        """Retourne l'action prédite (0-3). Fallback oracle si non entraîné."""
        if not self.trained or self.model is None:
            self.oracle.set_direction(self.direction)
            return self.oracle.get_action(state)

        x = np.array(state, dtype=np.float32).reshape(1, -1)
        x = self.scaler.transform(x)

        if self.use_cuda and XGBOOST_AVAILABLE:
            dmat   = xgb.DMatrix(x)
            action = int(self.model.predict(dmat)[0])
        else:
            action = int(self.model.predict(x)[0])

        # Sécurité : ne jamais aller à l'opposé de la direction actuelle
        current_dir = DIRECTION_MAP.get(self.direction, 1)
        if action == OPPOSITE[current_dir]:
            # Choisir parmi les autres actions les plus sûres
            state_arr = np.array(state)
            danger = {0: state_arr[0], 1: state_arr[2],
                      2: state_arr[4], 3: state_arr[6]}
            valid = {a: d for a, d in danger.items()
                     if a != OPPOSITE[current_dir]}
            action = max(valid, key=lambda a: valid[a])

        return action

    def set_direction(self, direction: str):
        """Mise à jour de la direction courante (appelée depuis main.py)."""
        self.direction = direction
        self.oracle.set_direction(direction)

    # ── Entraînement ──────────────────────────────────────────────────────
    def train(self, verbose: bool = True) -> float:
        """
        Entraîne (ou ré-entraîne) le modèle sur tout le buffer.
        Retourne le taux d'erreur de validation croisée (3-fold).
        """
        if len(self.buffer) < 200:
            print(f"[Train] Buffer trop petit ({len(self.buffer)} exemples). "
                  "Collecte plus de données d'abord.")
            return 1.0

        X, y = self.buffer.get_all()

        # Normalisation
        X_scaled = self.scaler.fit_transform(X)

        if verbose:
            print(f"[Train] Dataset : {len(X)} exemples | "
                  f"Distribution actions : "
                  f"{np.bincount(y, minlength=4)}")

        if self.use_cuda and XGBOOST_AVAILABLE:
            error_rate = self._train_xgboost(X_scaled, y, verbose)
        else:
            error_rate = self._train_sklearn(X_scaled, y, verbose)

        self.trained = True
        self.train_history.append({
            "n_samples":  len(X),
            "error_rate": error_rate,
        })
        if verbose:
            print(f"[Train] Erreur CV-3 : {error_rate:.4f}  "
                  f"(Accuracy ≈ {(1-error_rate)*100:.1f}%)")
        return error_rate

    def _train_xgboost(self, X: np.ndarray, y: np.ndarray,
                       verbose: bool) -> float:
        """Entraîne XGBoost avec CUDA."""
        params = XGBOOST_PARAMS.copy()
        if not self.use_cuda:
            params["device"] = "cpu"

        dtrain = xgb.DMatrix(X.astype(np.float32), label=y.astype(np.int32))

        # Early stopping sur 10% de validation
        n_val   = max(50, int(0.10 * len(X)))
        idx_val = np.random.choice(len(X), n_val, replace=False)
        mask    = np.ones(len(X), dtype=bool)
        mask[idx_val] = False

        dtrain_fit = xgb.DMatrix(X[mask].astype(np.float32),
                                  label=y[mask].astype(np.int32))
        dval       = xgb.DMatrix(X[idx_val].astype(np.float32),
                                  label=y[idx_val].astype(np.int32))

        evals       = [(dtrain_fit, "train"), (dval, "val")]
        evals_result = {}

        num_rounds = params.pop("n_estimators", 400)
        params.pop("use_label_encoder", None)
        params.pop("eval_metric",       None)

        self.model = xgb.train(
            params,
            dtrain_fit,
            num_boost_round   = num_rounds,
            evals             = evals,
            evals_result      = evals_result,
            early_stopping_rounds = 30,
            verbose_eval      = 50 if verbose else False,
        )

        # Calcul de l'erreur sur la validation
        preds      = self.model.predict(dval).astype(int)
        error_rate = 1.0 - accuracy_score(y[idx_val], preds)
        return error_rate

    def _train_sklearn(self, X: np.ndarray, y: np.ndarray,
                       verbose: bool) -> float:
        """Fallback GradientBoosting sklearn (CPU)."""
        if verbose:
            print("[Train] Utilisation de GradientBoostingClassifier (CPU).")
        self.model = GradientBoostingClassifier(**SKLEARN_PARAMS)
        scores     = cross_val_score(self.model, X, y, cv=3,
                                     scoring="accuracy", n_jobs=-1)
        self.model.fit(X, y)
        return 1.0 - float(scores.mean())

    # ── Collecte de données (oracle) ──────────────────────────────────────
    def record_step(self, state: List[float], action: int):
        """Enregistre manuellement un (state, action) dans le buffer."""
        self.buffer.push(state, action)

    def record_oracle_step(self, state: List[float]) -> int:
        """
        Fait jouer l'oracle sur `state`, enregistre, et retourne l'action.
        Utile pour la collecte automatique dans game_loop.
        """
        self.oracle.set_direction(self.direction)
        action = self.oracle.get_action(state)
        self.buffer.push(state, action)
        return action

    # ── Sauvegarde / chargement ───────────────────────────────────────────
    def save(self, model_path: Optional[str] = None,
             buffer_path: Optional[str] = None):
        mp = model_path or self.MODEL_PATH
        bp = buffer_path or self.BUFFER_PATH

        with open(mp, "wb") as f:
            pickle.dump({
                "model":         self.model,
                "scaler":        self.scaler,
                "trained":       self.trained,
                "train_history": self.train_history,
                "use_cuda":      self.use_cuda,
            }, f)
        print(f"[Save] Modèle sauvegardé dans {mp}")
        self.buffer.save(bp)

    def load(self, model_path: Optional[str] = None,
             buffer_path: Optional[str] = None):
        mp = model_path or self.MODEL_PATH
        bp = buffer_path or self.BUFFER_PATH

        if os.path.exists(mp):
            with open(mp, "rb") as f:
                data = pickle.load(f)
            self.model         = data["model"]
            self.scaler        = data["scaler"]
            self.trained       = data["trained"]
            self.train_history = data.get("train_history", [])
            print(f"[Load] Modèle chargé depuis {mp}")
        else:
            print(f"[Load] Aucun modèle trouvé à {mp}")

        if os.path.exists(bp):
            self.buffer.load(bp)
            # Validation : si les features du buffer ne correspondent plus, reset
            if len(self.buffer) > 0 and len(self.buffer.states[0]) != N_FEATURES:
                print(f"[Load] Feature mismatch : buffer={len(self.buffer.states[0])} "
                      f"vs attendu={N_FEATURES}. Reset buffer et modele.")
                self.buffer  = ReplayBuffer(max_size=300_000)
                self.trained = False
                self.model   = None

    # ── Statistiques ──────────────────────────────────────────────────────
    def stats(self):
        print("=" * 50)
        print(f"  Agent DecisionTree Snake")
        print(f"  Backend  : {'XGBoost CUDA' if self.use_cuda else 'Sklearn CPU'}")
        print(f"  Entraîné : {self.trained}")
        print(f"  Buffer   : {len(self.buffer)} exemples")
        if self.train_history:
            last = self.train_history[-1]
            print(f"  Dernière accuracy CV : "
                  f"{(1-last['error_rate'])*100:.1f}%  "
                  f"({last['n_samples']} samples)")
        print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# Interface Neat-compatible (pour ne pas modifier snake.py)
# ─────────────────────────────────────────────────────────────────────────────
class NeatCompatibleWrapper:
    """
    Wrapper qui expose exactement l'interface attendue par game_loop() :
      - tab_state(...)  → list[float]
      - get_action(net, state) → int
    `net` est ici l'instance de DecisionTreeAgent elle-même.
    """

    @staticmethod
    def tab_state(*args) -> List[float]:
        """Convertit les 16 distances en liste de floats."""
        return list(args)

    @staticmethod
    def get_action(agent: "DecisionTreeAgent",
                   state: List[float]) -> int:
        return agent.get_action(state)
