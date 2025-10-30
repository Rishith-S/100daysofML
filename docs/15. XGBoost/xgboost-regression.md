---
sidebar_position: 2
title: XGBoost Regression
---
## XGBoost for Regression: Notes and Cheatsheet

This page summarizes how XGBoost solves regression, from the objective to the split formulas you’ll actually tune, plus a minimal example and practical tips.

## Problem setup and objective

- Data: $D = \{(x_i, y_i)\}_{i=1}^n$
- Goal at boosting step $t$: learn a new tree $f_t(x)$ that improves predictions $\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta f_t(x)$
- Regularized objective minimized at step t:

$$
\mathcal{L}^{(t)} = \sum_{i=1}^{n} \ell\big(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\big) + \Omega(f_t)
$$

with tree penalty

$$
\Omega(f) = \gamma \cdot T + \tfrac{1}{2} \lambda \sum_{j=1}^{T} w_j^2 \;\;\text{(plus optional L1 with }\alpha\text{)}
$$

For squared error loss $\ell(y, \hat{y}) = \tfrac{1}{2}(y-\hat{y})^2$ the first and second derivatives are:

- Gradient: $g_i = \partial_{\hat{y}} \ell = (\hat{y}_i^{(t-1)} - y_i)$
- Hessian: $h_i = \partial^2_{\hat{y}} \ell = 1$

## Building one tree (second-order Taylor)

For a candidate tree structure, aggregate gradients and Hessians per leaf j:

- $G_j = \sum_{i \in I_j} g_i$
- $H_j = \sum_{i \in I_j} h_i$

Optimal leaf weight and the corresponding leaf score are:

$$
w_j^{\ast} = -\frac{G_j}{H_j + \lambda},\qquad \text{Score}(j) = -\tfrac{1}{2}\frac{G_j^2}{H_j + \lambda} + \gamma
$$

When evaluating a split into left (L) and right (R) leaves from a parent (P), the gain is:

$$
	ext{Gain} = \tfrac{1}{2} \Bigg( \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G_P^2}{H_P + \lambda} \Bigg) - \gamma
$$

- For squared error, $h_i=1$, so $H_j$ is just the number of samples in the leaf and $G_j$ is the sum of residuals.
- The algorithm chooses the split with the highest positive gain; if gain does not exceed $\gamma$, the split is pruned.

## Prediction update (shrinkage)

After fitting $f_t$, update predictions with learning rate $\eta$:

$$
\hat{y}^{(t)}(x) = \hat{y}^{(t-1)}(x) + \eta \cdot f_t(x)
$$

Smaller $\eta$ generally needs more trees (`n_estimators`) but improves generalization when paired with early stopping.

## Regularization and overfitting control

- Depth/structure: `max_depth`, `min_child_weight` (controls $H_j$), `gamma` (prunes low-gain splits)
- Shrinkage and size: `learning_rate` (η), `n_estimators` with `early_stopping_rounds`
- Sampling: `subsample` (rows), `colsample_bytree` and friends (columns)
- Weights: `reg_lambda` (L2), `reg_alpha` (L1)

## Handling missing values and categories

- Missing values are handled natively: each split learns a default direction for missing values that maximizes gain.
- For categorical features, classic XGBoost expects numeric encodings. Recent versions support `enable_categorical=True` with proper dtype, but one-hot or target/WOE encodings remain common in practice.

## Efficiency knobs

- `tree_method='hist'` for large data; `tree_method='gpu_hist'` for GPU acceleration
- Parallelization within a tree: set `n_jobs` (use -1 to utilize all CPU cores)

## Minimal example (scikit-learn API)

```python
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = fetch_california_housing(return_X_y=True, as_frame=False)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
	n_estimators=2000,
	learning_rate=0.03,
	max_depth=6,
	subsample=0.8,
	colsample_bytree=0.8,
	reg_lambda=1.0,
	reg_alpha=0.0,
	min_child_weight=1.0,
	tree_method="hist",
	n_jobs=-1,
	random_state=42,
)

model.fit(
	X_train, y_train,
	eval_set=[(X_valid, y_valid)],
	eval_metric="rmse",
	verbose=False,
	early_stopping_rounds=100,
)

pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, pred, squared=False)
print({"rmse": rmse, "best_iteration": model.best_iteration})
```

## A quick tuning recipe

- Start with `tree_method='hist'`, `learning_rate` around 0.05, `n_estimators` large (e.g., 2000–5000) and rely on `early_stopping_rounds`.
- Tune model size/complexity: `max_depth` 4–10, `min_child_weight` 1–10, `gamma` 0–5.
- Add randomness for robustness: `subsample` 0.6–1.0, `colsample_bytree` 0.6–1.0.
- Use `reg_lambda` (0.5–5) and optionally `reg_alpha` (0–1) to regularize.
- Monitor validation RMSE; prefer the iteration with the best score (early-stopping will keep it).

## Diagnostics and interpretation

- Feature importance: `model.get_booster().feature_names` and `model.feature_importances_` (gain-based) give a first glance; prefer permutation importance for reliability.
- SHAP values provide local and global explanations for tree ensembles and are well-supported with XGBoost.

## Common pitfalls

- Too-large `learning_rate` with too-few trees overfits quickly; prefer small `learning_rate` with early stopping.
- Data leakage from preprocessing/target leakage can make validation scores optimistic; use proper cross-validation.
- Unscaled targets with huge variance can slow learning; consider target transforms where appropriate.
- Highly skewed loss due to outliers: try Huber/Quantile losses via custom objectives or robust preprocessing.