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

## ⏱️ Training Time (estimated)

Grid: `800×400 px` — cells `50×50` → **16×8 = 128 cells**.
No display during training (`SHOW_GAME = False`) → ~0.3–1 ms per game step (pure Python).

| Phase | Steps | GPU (CUDA) | CPU (sklearn fallback) |
|-------|-------|------------|------------------------|
| Phase 1 — Oracle | 500 games × ~400 steps avg = ~200k steps | ~2–3 min | ~2–3 min |
| Phase 2 — Initial training | ~200k samples, 400 rounds, early stop @30 | ~10–20 s | ~3–8 min |
| Phase 3 — DAgger | 400 games (~160k steps) + 4 retrainings on ~300k samples | ~3–5 min | ~15–25 min |
| Phase 4 — Evaluation | 100 games × ~300 steps | ~30–60 s | ~30–60 s |
| **Total** | | **~6–10 min** | **~20–40 min** |

---

## 📈 Performance

| Metric | Value | Source |
|--------|-------|--------|
| **Score max observed** | **31** | Visible in the demo GIF |
| Score mean (estimated) | 5–15 | Not logged — derived from oracle baseline on 16×8 grid |
| Oracle greedy baseline | ~8–20 mean | Greedy heuristic with enriched features (danger binary + food delta) |
| Grid cells | 128 | 16 columns × 8 rows |

> ⚠️ Mean score is an estimate — no score logging is implemented in the current pipeline.
> Run `python main.py` and read the Phase 4 output to get the exact value.
