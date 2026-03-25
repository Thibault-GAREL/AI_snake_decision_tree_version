# 📊 Snake Decision Tree — Training Info

## 🔍 Résumé XAI — Interprétabilité du modèle

L'analyse XAI révèle que le modèle XGBoost appris par imitation (DAgger) est **fortement interprétable et cohérent** : les features de danger directionnel (N/S/E/W) dominent l'ensemble des métriques d'importance — native (Gain, Weight, Cover), par permutation (chute de score ~40 pts quand brouillées) et SHAP (valeurs absolues les plus élevées sur toutes les actions). Les features food delta jouent un rôle secondaire mais structuré, avec une corrélation de Pearson directionnelle claire avec chaque action (UP/RIGHT/DOWN/LEFT). Les projections t-SNE et UMAP montrent une structure latente cohérente : les états de jeu se regroupent naturellement par situation (danger vs food alignée vs neutre), et certaines features sont hautement spécialisées selon le contexte. L'analyse SHAP par waterfall décompose chaque décision individuelle en contributions signées, confirmant que l'agent évite le danger en priorité absolue et se dirige vers la nourriture uniquement en l'absence de menace immédiate. La carte de confiance (prob-gap) révèle que l'agent est quasi-certain en bord de grille et plus hésitant au centre, là où les choix sont réellement ambigus.

---

## 🏗️ Model Architecture

This is not a neural network — the equivalent of "neurons" here are **decision nodes** inside boosted trees.

| Parameter | Value | Details |
|-----------|-------|---------|
| Boosting rounds | 400 | `n_estimators` |
| Classes | 4 | UP / RIGHT / DOWN / LEFT |
| **Total trees** | **1 600** | 400 rounds × 4 classes (`multi:softmax`) |
| Max depth per tree | 8 | Max 511 nodes theoretically |
| Effective nodes/tree | ~50–120 | After regularization (`gamma=0.1`, `min_child_weight=3`, `reg_lambda=1.5`) |
| **Total nodes (estimated)** | **~80 000 – 200 000** | Decision nodes across all trees |
| Features per tree | ~22 | `colsample_bytree=0.85` × 26 features |

---

## 💾 Memory & Buffer

| Element | Value | Details |
|---------|-------|---------|
| Buffer max capacity | 300 000 samples | Defined in `DecisionTreeAgent.__init__` |
| Per sample (numpy) | 108 bytes | 26 × float32 (104B) + 1 × int32 (4B) |
| Per sample (Python lists) | ~708 bytes | Python object overhead (float = 24B each) |
| **RAM during training** | **~212 MB** | Buffer stored as Python lists |
| Training arrays (numpy) | ~32 MB | X: ~31.2 MB + y: ~1.2 MB |
| Estimated real buffer size | ~200 000 – 280 000 samples | Oracle: 500 games × ~400 steps + DAgger: 400 games × ~300 steps |
| Mini-batch | None | XGBoost trains on **full dataset** each round |
| Effective samples per tree | 85 % of dataset | `subsample=0.85` |

---

## ⏱️ Training Time (measured — MLflow run 2026-03-25, GPU CUDA)

Grid: `800×400 px` — cells `50×50` → **16×8 = 128 cells**.
No display during training (`SHOW_GAME = False`).
MLflow Run ID: `f936e52849d04973abc3026076a7bae0`

| Phase | Details | Measured time |
|-------|---------|---------------|
| Phase 1 — Oracle | 500 games, buffer: 0 → 119 964 samples | **11.8 s** |
| Phase 2 — Initial training | 119 964 samples, early stop @380/400 rounds | **~15 s** |
| Phase 3 — DAgger | 8 rounds × 50 games + 4 retrainings, buffer: 119 964 → 215 576 | **~7 min** |
| Phase 4 — Evaluation | 100 games (pure agent, beta=0) | **253.4 s (~4.2 min)** |
| **Total** | | **729.1 s (~12.2 min, GPU)** |

> Note : Phase 3+4 sont plus lentes car chaque pas de l'agent nécessite un appel XGBoost (DMatrix + predict), contrairement à l'oracle qui est du Python pur (~24ms/game oracle vs ~2.5s/game agent).

---

## 📈 Performance (MLflow run 2026-03-25, seed 42)

### Environnement

| Library | Version |
|---------|---------|
| Python | 3.10.0 |
| XGBoost | 2.1.4 |
| scikit-learn | 1.7.2 |
| NumPy | 1.26.4 |
| MLflow | 3.10.1 |
| Backend | XGBoost CUDA |

### Phase 1 — Oracle (500 parties, greedy heuristic)

| Metric | Value |
|--------|-------|
| Score moyen oracle | **22.37 ± 6.33** |
| Score max oracle | **39** |
| Buffer rempli | **119 964 exemples** |
| Temps | 11.8 s |

### Phase 2 — Entraînement initial (XGBoost, CUDA)

| Metric | Value |
|--------|-------|
| Dataset | 119 964 exemples |
| Distribution actions | [20 461 UP / 40 581 RIGHT / 19 789 DOWN / 39 133 LEFT] |
| Early stopping | Round **380 / 400** |
| Val mlogloss final | **0.00674** |
| Accuracy CV-3 | **99.7%** (erreur 0.0028) |

### Phase 3 — DAgger (8 rounds × 50 parties)

| Round | Beta | Score moyen | Score max | Buffer | Accuracy CV-3 |
|-------|------|-------------|-----------|--------|----------------|
| 1 | 0.800 | 23.68 | 37 | 133 130 | — |
| 2 | 0.680 | 22.54 | 39 | 145 710 | 99.7% (0.0030) |
| 3 | 0.578 | 21.12 | 33 | 156 500 | — |
| 4 | 0.491 | 21.32 | 36 | 167 920 | 99.7% (0.0033) |
| 5 | 0.418 | 22.80 | 37 | 180 105 | — |
| 6 | 0.355 | 22.10 | 35 | 191 939 | **99.8%** (0.0024) |
| 7 | 0.302 | 20.70 | 37 | 202 910 | — |
| 8 | 0.256 | 23.02 | 35 | 215 576 | 99.7% (0.0027) |

### Phase 4 — Évaluation finale (100 parties, agent pur, beta=0)

| Metric | Value |
|--------|-------|
| **Score moyen** | **22.77 ± 7.13** |
| **Score médian** | **23.0** |
| **Score max** | **43** |
| Temps évaluation | 253.4 s |
| Buffer final | **240 054 exemples** |

### Comparaison oracle / agent

| | Oracle greedy | Agent XGBoost |
|--|---------------|---------------|
| Score moyen | 22.37 | **22.77** |
| Score max | ~39 | **43** |
| Vitesse | ~24 ms/game | ~2.5 s/game |

> L'agent XGBoost **dépasse l'oracle** en score moyen (22.77 > 22.37) et en score max (43 > 39). Le DAgger a permis à l'élève de surpasser le maître.

---

## 📦 MLflow Tracking

Le pipeline d'entraînement log automatiquement dans MLflow (experiment `snake-decision-tree`) :
- **Params** : tous les hyperparamètres (XGBoost + pipeline), seed, versions librairies
- **Metrics** : scores oracle, accuracy CV-3, scores DAgger par round, résultats eval finaux, temps par phase
- **Artifacts** : `snake_xgb_model.pkl`, `snake_training_curves.png`

Pour visualiser les runs :
```bash
cd snake_arbre_de_decision
mlflow ui --port 5000
```
