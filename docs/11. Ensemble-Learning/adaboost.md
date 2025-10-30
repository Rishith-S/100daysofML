---
sidebar_position: 6
title: AdaBoost — Intuition, Mechanics, and Tuning
---

## Overview

AdaBoost (Adaptive Boosting) is a boosting algorithm that builds a strong classifier by combining many weak learners in sequence. Each learner focuses more on the training points the previous learners struggled with. Understanding AdaBoost gives a solid foundation for other boosting methods like Gradient Boosting, XGBoost, and LightGBM.

## Core concepts

- Weak learner (base estimator): typically a Decision Tree stump (max_depth = 1). Stumps split once on a single feature to maximise impurity reduction.
- Labels: often encoded as −1 and +1 in the binary case.
- Weighted error (err): sum of sample weights of misclassified points for the current learner.
- Learner weight (alpha): contribution of a learner in the final ensemble. Better learners (lower error) receive higher alpha.
- Sample reweighting (boosting): increase weights of misclassified points and decrease weights of correctly classified points; then renormalise so weights sum to 1.

## How AdaBoost works (binary intuition)

Given N training points with initial weights $w_i^{(1)} = 1/N$.

Repeat for t = 1, 2, …, T:

1) Fit a weak learner $h_t(x)$ using the current sample weights.

2) Compute weighted error

$$\operatorname{err}_t = \sum_{i=1}^{N} w_i^{(t)} \,[\, y_i \ne h_t(x_i) \,]$$

3) Compute learner weight (SAMME, binary)

$$\alpha_t = \tfrac{1}{2} \ln \frac{1 - \operatorname{err}_t}{\operatorname{err}_t}$$

4) Update sample weights

$$w_i^{(t+1)} = \frac{w_i^{(t)} \exp\big( -\alpha_t \, y_i \, h_t(x_i) \big)}{Z_t}$$

where $Z_t$ normalises the weights to sum to 1, and $y_i, h_t(x_i) \in \{-1, +1\}$.

Final prediction for a new x:

$$\hat{y} = \operatorname{sign}\Big( \sum_{t=1}^{T} \alpha_t \, h_t(x) \Big).$$

Notes on multi‑class

- SAMME (discrete): generalises the above; $\alpha_t$ depends on error and number of classes K via $\alpha_t = \ln\frac{1-\operatorname{err}_t}{\operatorname{err}_t} + \ln(K-1)$.
- SAMME.R (real): uses class probabilities from the weak learner and typically converges faster; default in scikit‑learn.

## Why it works

Each learner concentrates on the “hard” points by increasing their weights. The weighted vote aggregates diverse decision stumps into a piece‑wise nonlinear boundary with low training bias, while staged, small updates (optionally via a learning rate) control variance and improve generalisation.

## Practical defaults (scikit‑learn)

- Base estimator: `DecisionTreeClassifier(max_depth=1)` (stump) is standard and very effective.
- Number of estimators: `n_estimators` in the range 50–400 is common; too few underfit, too many may overfit and slow down training.
- Learning rate: scales the stage contribution. Smaller values (for example 0.1) slow learning and can reduce overfitting; pair with a larger `n_estimators`.
- Algorithm: `"SAMME.R"` (default) vs `"SAMME"`.
- Random state: set for reproducibility.

## Classification example

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

stump = DecisionTreeClassifier(max_depth=1, random_state=42)
ada = AdaBoostClassifier(
	estimator=stump,        # scikit‑learn >= 1.2 (formerly base_estimator)
	n_estimators=200,
	learning_rate=0.5,
	algorithm="SAMME.R",
	random_state=42,
)

ada.fit(X_train, y_train)
print("Test accuracy:", accuracy_score(y_test, ada.predict(X_test)))
```

Tips

- If your base estimator does not support `predict_proba`, prefer `algorithm="SAMME"` (discrete). With probability support, `SAMME.R` is usually stronger.
- Use `staged_predict` or `staged_predict_proba` to plot validation performance versus number of estimators and pick a good stopping point.

## Regression variant (AdaBoostRegressor)

AdaBoost also works for regression by reweighting observations using a loss‑based scheme. Default base estimator is a shallow tree regressor.

```python
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

base = DecisionTreeRegressor(max_depth=3, random_state=42)
ada_r = AdaBoostRegressor(
	estimator=base,
	n_estimators=300,
	learning_rate=0.1,
	random_state=42,
)

ada_r.fit(X_train, y_train)
y_pred = ada_r.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse:.3f}")
```

## Hyperparameters to tune

- `n_estimators`: increases model capacity; monitor validation curves to avoid overfitting.
- `learning_rate`: shrink each stage’s impact. Lower values usually require higher `n_estimators`.
- `estimator` hyperparameters: e.g., tree depth; depth 1 (stump) is classic for AdaBoostClassifier.
- `algorithm`: try `SAMME` if base estimators lack calibrated probabilities or if you want discrete boosting.

### Quick search patterns

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
	AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=42), random_state=42),
	param_grid={
		"n_estimators": [50, 100, 200, 400],
		"learning_rate": [0.05, 0.1, 0.2, 0.5, 1.0],
		"algorithm": ["SAMME.R", "SAMME"],
	},
	cv=5,
	n_jobs=-1,
)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```

## Summary

- AdaBoost adds weak learners sequentially, focusing on previously misclassified points via weight updates.
- Learner weight (alpha) increases as weighted error decreases; final decision is a weighted vote.
- Tune `learning_rate` and `n_estimators` together; prefer decision stumps as base estimators for classification.
- For multi‑class tasks, use `SAMME.R` when probabilities are available; fall back to `SAMME` otherwise.

