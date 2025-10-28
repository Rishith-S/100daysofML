---
title: Elastic Net Regression
sidebar_position: 1
---

Elastic Net is a linear model that blends Ridge (L2) and Lasso (L1) penalties. It’s handy when you have many features, possible multicollinearity, and you’re unsure whether pure Ridge or pure Lasso is best.

## What problem it solves
- You’ve got a large feature set and don’t know which inputs are important.
- Your inputs are correlated (multicollinearity), e.g., height and weight.
- You want a compromise between Ridge’s stability and Lasso’s feature selection.

## Loss function (safe text)
`Loss = MSE + A * (Σ w_i^2) + B * (Σ |w_i|)`

- `A` is the weight on the Ridge (L2) penalty
- `B` is the weight on the Lasso (L1) penalty

In scikit‑learn:
- `alpha` ≈ overall regularization strength: `alpha = A + B`
- `l1_ratio` sets the L1/L2 mix: `l1_ratio = A / (A + B)`
  - So `A = l1_ratio * alpha`
  - `B = (1 - l1_ratio) * alpha`
- Defaults: `alpha = 1.0`, `l1_ratio = 0.5` (equal L1 and L2 weight)
- Set `l1_ratio` closer to 1.0 → more L1 behavior (sparser)
- Set `l1_ratio` closer to 0.0 → more L2 behavior (more stable with collinearity)

## Ridge vs Lasso context
- Ridge (L2): shrinks all coefficients toward zero but rarely to exactly zero; good with multicollinearity; keeps all features.
- Lasso (L1): can set some coefficients exactly to zero → feature selection; may be unstable when predictors are highly correlated.
- Elastic Net: balances both; tends to keep groups of correlated features while still enabling sparsity.

## Practical guidance
- Scale features (standardization) before fitting; penalties are scale‑dependent.
- Tune both `alpha` and `l1_ratio` via cross‑validation.
- Try Linear Regression, Ridge, Lasso, and Elastic Net on wide datasets (e.g., 50+ columns) and compare validation metrics.

## Observations (from practice)
- On some multicollinear datasets, Ridge can beat Lasso slightly.
- Elastic Net with, for example, `alpha=0.005` and `l1_ratio=0.9` (more Ridge weight) may outperform both individually—tune on your data to be sure.

## Minimal scikit‑learn examples

### Direct fit
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42))
model.fit(X_train, y_train)
print("R2 on validation:", model.score(X_val, y_val))
```

### With cross‑validated tuning
```python
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("enet", ElasticNet(max_iter=10000, random_state=42))
])

param_grid = {
    "enet__alpha": [0.001, 0.005, 0.01, 0.1, 1.0],
    "enet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
}

search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
print("Best CV R2:", search.best_score_)
```

### SGDRegressor alternative
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline

sgd_enet = make_pipeline(
    StandardScaler(),
    SGDRegressor(penalty="elasticnet", alpha=0.001, l1_ratio=0.5, max_iter=2000, tol=1e-3, random_state=42)
)
sgd_enet.fit(X_train, y_train)
```

## Bias–variance intuition
- Increasing `alpha` increases bias and reduces variance.
- `l1_ratio` controls sparsity vs. stability: higher → sparser (more L1), lower → more stable with correlated features (more L2).

## Key takeaways
1. Elastic Net = L1 + L2: one knob for strength (`alpha`), one knob for mix (`l1_ratio`).
2. Great when features are many and correlated.
3. Tune both hyperparameters; standardize features first.
4. Often more robust than pure Lasso in the presence of multicollinearity while still enabling sparsity.
