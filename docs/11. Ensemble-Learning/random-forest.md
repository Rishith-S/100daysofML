---
sidebar_position: 4
title: Random Forest — Intuition, Differences from Bagging, and Hyperparameters
---

## Why Random Forest?

Random Forest is a flexible, high‑performing ensemble that works well out‑of‑the‑box on many tabular tasks for both classification and regression. It’s essentially Bagging with Decision Trees as base learners plus extra randomness that further reduces variance.

## Bagging vs Random Forest — What’s the difference?

1) Base algorithm

- Bagging: generic — can use many algorithms (Decision Trees, KNN, SVM, …).
- Random Forest: trees only. Every base learner is a Decision Tree.

2) Feature (column) sampling strategy

- Bagging with trees (generic bagging): if you sub‑sample features, the subset is typically chosen once per base model (tree‑level). The tree is trained using only that feature subset for all its splits.
- Random Forest: sub‑sample features at every node (split‑level). For each split, a new random subset of features is considered. This injects more randomness, de‑correlates trees, and usually improves performance vs tree‑level subspaces.

Result: RF ≈ Bagging(Decision Trees) + node‑level feature sampling. More de‑correlated trees → stronger averaging → lower variance.

## Bias–variance intuition: why RF performs so well

- Single fully grown trees: low bias, high variance (overfit; sensitive to data quirks/outliers).
- RF trains many such trees on bootstrapped samples and random feature subsets, then aggregates:
	- Bootstrapping spreads noise/outliers across trees.
	- Node‑level feature sampling reduces correlation among trees.
	- Averaging votes (classification) or predictions (regression) cancels variance while keeping bias low → low bias, low variance.

Visual intuition

- Classification: single tree has jagged, overfit boundaries; RF yields smoother, more stable boundaries.
- Regression: single tree produces a step‑like, wiggly curve; RF averages these into a smoother function with lower generalisation error.

## How Random Forest works (high level)

1. Build T Decision Trees. For each tree:
	 - Draw a bootstrap sample of rows (sampling with replacement).
	 - Grow the tree to (near) purity without pruning.
	 - At each split, sample a subset of features of size `max_features` and choose the best split among those.
2. Aggregate predictions over all trees:
	 - Classification: majority vote (or average probabilities).
	 - Regression: mean of predictions.

### Out‑of‑Bag (OOB) estimate

With bootstrapping, about 36.8% of training rows are OOB for any given tree (probability of not being chosen: $(1 - 1/N)^N \approx e^{-1}$). RF can evaluate these OOB rows to provide an internal validation score via `oob_score=True`.

## Key hyperparameters (scikit‑learn)

Random‑forest specific

- `n_estimators` (default ~100): number of trees. More → more stable up to diminishing returns.
- `max_features`: features to consider at each split. Options: integer, float fraction, `'sqrt'`, `'log2'`, or `None` (all features). Common choices:
	- Classification default: `'sqrt'`.
	- Regression typical: `1.0` (all) or a fraction; try `'sqrt'`/`'log2'` for extra decorrelation.
- `bootstrap` (default True): sample rows with replacement.
- `max_samples` (if `bootstrap=True`): number or fraction of rows per tree. Often 0.5–0.75 works well.
- `oob_score`: enable OOB evaluation.

Decision‑tree inherited

- `criterion`: split quality. Classifier: `"gini"`, `"entropy"`/`"log_loss"`. Regressor: `"squared_error"` (formerly "mse"), `"absolute_error"` (formerly "mae"), `"poisson"`.
- `max_depth`: limit depth to curb overfitting (or leave `None` to grow deep).
- `min_samples_split`, `min_samples_leaf`: increase to reduce variance; `min_samples_leaf` is a strong regulariser.
- `max_leaf_nodes`, `min_impurity_decrease`: additional regularisation/pruning controls.
- `ccp_alpha`: minimal cost‑complexity pruning parameter.

General

- `n_jobs`: parallelism. Use `-1` to utilise all cores.
- `random_state`: reproducibility of sampling/feature subsampling.
- `class_weight` (classifier): handle class imbalance (`"balanced"`).
- `verbose`, `warm_start`: diagnostics and incremental growth.

## Minimal examples

Classification

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
		n_estimators=200,
		max_features="sqrt",   # default for classifier
		bootstrap=True,
		oob_score=True,
		n_jobs=-1,
		random_state=42,
)

rf.fit(X_train, y_train)
print("OOB score:", getattr(rf, "oob_score_", None))
print("Test accuracy:", accuracy_score(y_test, rf.predict(X_test)))
```

Regression

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rf_reg = RandomForestRegressor(
		n_estimators=300,
		max_features=1.0,       # try "sqrt" / "log2" as well
		bootstrap=True,
		oob_score=True,
		n_jobs=-1,
		random_state=42,
)

rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse:.3f}")
```

## Tuning tips

- Start with `n_estimators` ∈ [100, 500]; increase until OOB/CV score stabilises.
- Try `max_features` in ("sqrt", "log2", or fractions like 0.3–0.8). Smaller values → more randomness (lower correlation) but may raise bias; find the balance with CV.
- Consider setting `min_samples_leaf` to 1–10 to stabilise leaves; helpful on noisy data.
- Use `max_samples` < 1.0 to add diversity (e.g., 0.6–0.8) if datasets are large.
- For imbalanced classes, set `class_weight="balanced"` and evaluate with appropriate metrics (AUC, F1).

## Quick recap: interview‑ready differences

- Bagging: generic template; any base algorithm. Feature subspace often chosen per base model.
- Random Forest: trees only; feature subspace chosen independently at every node (split). More randomness → lower inter‑tree correlation → stronger averaging → better generalisation.

