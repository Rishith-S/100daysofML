---
sidebar_position: 1
title: Gradient Boosting — Regression
---

## What is Gradient Boosting?

Gradient Boosting builds a strong predictor by adding many small regression trees sequentially. Each new tree is trained to correct the mistakes of the current model by fitting the negative gradient (pseudo‑residuals) of a chosen loss. It works extremely well on tabular data and supports both regression and classification.

## Core intuition for regression

1) Additive modelling

- Build the model as a sum of simple functions: a constant plus many shallow trees.

2) First model F₀(x)

- For squared error, start with the mean of the target: $F_0(x) = \operatorname{argmin}_\gamma \sum_i (y_i - \gamma)^2 = \overline{y}$.

3) Pseudo‑residuals

- For a differentiable loss $L(y, F)$, define residuals at stage m as the negative gradient with respect to the current prediction:

	$$ r_{im} = -\left. \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right\rvert_{F=F_{m-1}}. $$

- With squared error $\tfrac{1}{2}(y-\hat{y})^2$, this simplifies to $r_{im} = y_i - F_{m-1}(x_i)$.

4) Fit a tree to residuals

- Train a small regression tree $h_m(x)$ (e.g., depth 2–4 or with 8–32 leaf nodes) to predict $r_{im}$ from the original features.

5) Leaf values

- For each leaf region $R_{jm}$, compute an optimal value

	$$ \gamma_{jm} = \operatorname{argmin}_\gamma \sum_{x_i \in R_{jm}} L\big(y_i, F_{m-1}(x_i) + \gamma \big). $$

- For squared error, $\gamma_{jm}$ is the mean of residuals in that leaf.

6) Update the model (shrinkage with learning rate $\eta$)

	$$ F_m(x) = F_{m-1}(x) + \eta \sum_j \gamma_{jm} \mathbf{1}\{x \in R_{jm}\}. $$

Repeat steps 3–6 for $m = 1,\dots,M$.

## Why it works

- Sequentially fits what remains unexplained (the gradient of loss), reducing bias step by step.
- Using shallow trees keeps each step simple; shrinkage (small learning rate) and limited tree size control variance.

## Differences from AdaBoost (quick view)

- AdaBoost reweights samples based on misclassification; Gradient Boosting fits gradients of a general loss.
- AdaBoost commonly uses decision stumps; Gradient Boosting uses small regression trees with several leaves.
- AdaBoost assigns per‑stage weights (alphas); Gradient Boosting scales every stage with the same learning rate $\eta$ and optimised leaf values.

## scikit‑learn example

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(
		loss="squared_error",   # or "absolute_error", "huber", "quantile"
		learning_rate=0.1,       # shrinkage
		n_estimators=300,        # number of boosting stages
		max_depth=3,             # depth of individual trees
		subsample=1.0,           # <1.0 = stochastic gradient boosting
		max_features=None,       # try "sqrt"/"log2" or fractions on wide data
		random_state=42,
)

gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse:.3f}")
```

### Monitoring learning

You can probe validation performance as stages are added and pick an early‑stopping point:

```python
import numpy as np
from sklearn.metrics import mean_squared_error

val_rmse = []
for y_stage in gbr.staged_predict(X_test):
		val_rmse.append(mean_squared_error(y_test, y_stage, squared=False))

best_iter = int(np.argmin(val_rmse)) + 1
print("Best number of stages:", best_iter)
```

## Key hyperparameters (and typical ranges)

- learning_rate (0.01–0.2): smaller values need larger `n_estimators` but generalise better.
- n_estimators (100–1000): more stages increase capacity; pair with learning_rate.
- max_depth (2–5) or max_leaf_nodes (8–32): controls tree complexity per stage.
- subsample (0.5–1.0): values less than 1.0 enable stochastic gradient boosting and help reduce variance.
- max_features: limit features per split to add randomness on wide datasets.
- min_samples_leaf (1–20): stabilises leaves; helps with noisy targets.
- loss: `squared_error`, `absolute_error` (MAE), `huber` (robust), `quantile` (for pinball/quantile regression).

## Practical tips

- Start with `learning_rate=0.1`, `max_depth=3`, `n_estimators` around 300; tune with CV.
- Prefer `subsample` in 0.6–0.9 on larger datasets for extra regularisation.
- Watch for overfitting: track validation with `staged_predict`; use `n_iter_no_change` and `validation_fraction` for built‑in early stopping.
- Standardise or robust‑scale only if features differ wildly in scale; trees are generally scale‑insensitive.

## Summary

- Gradient Boosting = fit shallow trees to the gradient of the loss and add them with shrinkage.
- Excellent on tabular regression; robust options (`absolute_error`/`huber`) handle outliers.
- Balance learning_rate and n_estimators; control tree size and use subsampling to regularise.

