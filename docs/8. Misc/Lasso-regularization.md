---
sidebar_position: 4
---

# Lasso (L1) Regularization

Lasso is Linear Regression with an L1 penalty on the weights. The penalty pushes many coefficients exactly to zero, creating a sparse model and doubling as embedded feature selection.

- Loss function (safe text): `Loss = MSE + λ Σ |w_i|`
- λ (alpha) controls the strength of regularization.
  - `λ = 0` → Ordinary Least Squares (no regularization)
  - Large `λ` → Coefficients shrink strongly; many become `0` (all can become `0` in the limit)

## Why Lasso sets weights to exactly zero
- The L1 constraint forms a diamond-shaped region; the optimum often lands on a corner → one or more coefficients exactly `0`.
- Optimization uses soft-thresholding/coordinate descent: small coefficients are “thresholded” to zero.
- Result: sparse solutions → built‑in feature selection, simpler interpretation.

## Effect on coefficients
- Encourages sparsity: many `w_i = 0`.
- Among correlated features, Lasso tends to pick one and zero out the rest (can be unstable across samples).
- Intercept is not penalized by default.

## Bias–variance impact
- Increasing `λ`:
  - Increases bias (underfitting risk)
  - Decreases variance (better generalization, less overfitting)
- Choose `λ` via cross‑validation to balance bias and variance.

## When to use Lasso
- High‑dimensional data (p ≫ n) where feature selection is valuable.
- You want a compact, interpretable model with few non‑zero coefficients.
- Many weak/irrelevant features are present.

## When to be careful
- Strongly correlated predictors: Lasso’s choice among them can be unstable.
- If you need to keep groups of correlated features together, consider Elastic Net instead.

## Lasso vs Ridge vs Elastic Net
- Ridge (L2): `Loss = MSE + λ Σ w_i^2` → shrinks coefficients toward 0 but rarely exactly 0; good with multicollinearity; spreads weight across correlated features.
- Lasso (L1): drives many coefficients to 0 → feature selection, but unstable with highly correlated features.
- Elastic Net (α mix of L1/L2): combines benefits; tends to keep groups of correlated features; often a robust default.

## Practical tips
- Scale features (standardization) before applying Lasso; penalties are scale‑dependent.
- Tune `λ` (often called `alpha`) with cross‑validation.
- Check coefficient paths as `λ` varies to understand model stability.
- For categorical features with one‑hot encoding, keep a consistent reference and consider group penalties if needed.

## Tiny contract
- Inputs: numeric features X (scaled), target y, hyperparameter `λ`.
- Output: weight vector w with many zeros, intercept b.
- Success: low validation error with sparse, interpretable coefficients.
- Error modes: too large `λ` → all zeros (underfit); too small `λ` → overfit; correlated features → unstable selection.

## Minimal sklearn example
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline

# Pipeline scales features then runs Lasso with CV to pick λ (alpha)
model = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42))
model.fit(X_train, y_train)

print("Chosen alpha:", model.named_steps['lassocv'].alpha_)
coef = model.named_steps['lassocv'].coef_
print("Non-zero features:", (coef != 0).sum())
```

## Key takeaways
1. Lasso adds an L1 penalty: `Loss = MSE + λ Σ |w_i|`.
2. It can make coefficients exactly zero → automatic feature selection.
3. Tune `λ` via CV; scale features first.
4. Prefer Elastic Net when features are strongly correlated.
