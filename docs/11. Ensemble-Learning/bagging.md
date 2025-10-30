---
sidebar_position: 3
title: Bagging (Bootstrap Aggregation)
---

## Part 1: Bagging — Introduction

Bagging (Bootstrap Aggregation) is a core ensemble method alongside Boosting and Stacking. It reduces variance by training many versions of a low‑bias, high‑variance model on resampled datasets and aggregating their predictions.

### How Bagging works

1) Bootstrap (data resampling)

- Build multiple base models (M1, M2, …) of the same algorithm (e.g., all Decision Trees, all SVMs).
- For each model, draw a bootstrap sample: sample training rows with replacement from the original dataset. Some rows repeat; some are left out.
- Train each base model independently on its own bootstrap sample.

2) Aggregation (combine predictions)

- Classification: majority vote (or probability average).
- Regression: arithmetic mean of predictions.

### Why Bagging helps (bias–variance)

- Many strong base learners (e.g., deep trees, KNN, SVM) have low bias but high variance: they fit training data well but fluctuate across datasets.
- Training them on varied bootstrap samples de‑correlates their errors. Aggregation then cancels noise, yielding lower variance and better generalisation — approaching the ideal: low bias and low variance.

## When to use Bagging

- As a default try for high‑variance learners (Decision Trees, KNN, SVM).
- Works with most estimators, not just trees. Random Forest is a specialised Bagging of trees with extra feature randomness.

## Variants by sampling scheme

- Bagging: row sampling with replacement (standard approach).
- Pasting: row sampling without replacement.
- Random Subspaces: column (feature) sampling only; no row sampling — helpful with many features.
- Random Patches: sample both rows and columns — useful in high‑dimensional settings.

### Out‑of‑Bag (OOB) evaluation

For each bootstrap, the probability a given row is not selected is $(1 - 1/N)^N \approx e^{-1} \approx 0.368$. Thus about 36.8% of rows are OOB per model. You can use OOB samples as a built‑in validation set by enabling `oob_score=True`.

## Part 2: BaggingClassifier

Key parameters (scikit‑learn):

- `estimator`: base estimator (e.g., `DecisionTreeClassifier()`), formerly `base_estimator`.
- `n_estimators`: number of base models.
- `max_samples`: number (int) or fraction (float) of rows per base model; common range 0.25–0.5.
- `bootstrap`: whether to sample rows with replacement (Bagging) vs without (Pasting).
- `max_features`: number or fraction of features per base model.
- `bootstrap_features`: whether to sample features with replacement.
- `oob_score`: compute OOB performance if `bootstrap=True`.
- `n_jobs`: parallelism; use `-1` to utilise all cores.

### Example (classification)

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

base = DecisionTreeClassifier(max_depth=None, random_state=42)
bag = BaggingClassifier(
	estimator=base,          # scikit‑learn >= 1.2
	n_estimators=100,
	max_samples=0.5,
	max_features=1.0,
	bootstrap=True,
	oob_score=True,
	n_jobs=-1,
	random_state=42,
)

bag.fit(X_train, y_train)
print("OOB score:", getattr(bag, "oob_score_", None))
y_pred = bag.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
```

Comparison intuition:

- Single Decision Tree: often overfits, complex boundaries, high variance.
- BaggingClassifier: smoother effective boundary, better test accuracy via reduced variance.

## Tips and learnings (classification)

- Bagging typically outperforms Pasting due to additional randomness and de‑correlation.
- Try `max_samples` in 0.25–0.5 when row sampling; tune with CV.
- Use Random Subspaces or Random Patches on very wide datasets (many features).
- Remove weak base models; diversity helps only if learners are at least decent.

## Part 3: BaggingRegressor

Core idea mirrors classification; aggregation is the mean of base predictions.

Key parameters are analogous to `BaggingClassifier`:

- `estimator` (e.g., `DecisionTreeRegressor()`), `n_estimators`, `max_samples`, `max_features`, `bootstrap`, `bootstrap_features`, `oob_score`, `n_jobs`.

### Example (regression)

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

base = DecisionTreeRegressor(random_state=42)
bag = BaggingRegressor(
	estimator=base,
	n_estimators=80,
	max_samples=1.0,
	max_features=1.0,
	bootstrap=True,
	oob_score=True,
	n_jobs=-1,
	random_state=42,
)

bag.fit(X_train, y_train)
print("OOB score:", getattr(bag, "oob_score_", None))
y_pred = bag.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
```

## Hyperparameter tuning

Use cross‑validated search (Grid or Randomized) to tune:

- `n_estimators` (more improves stability up to a point)
- `max_samples` (row fraction)
- `max_features` (feature fraction)
- `bootstrap` / `bootstrap_features`
- Base estimator hyperparameters (e.g., depth of tree, K in KNN)

Example grid (classification):

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag = BaggingClassifier(
	estimator=DecisionTreeClassifier(random_state=42),
	random_state=42,
	n_jobs=-1,
)

param_grid = {
	"n_estimators": [50, 100, 200],
	"max_samples": [0.25, 0.5, 0.75, 1.0],
	"max_features": [0.5, 0.75, 1.0],
	"bootstrap": [True, False],
}

grid = GridSearchCV(bag, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```

## Quick takeaways

- Bagging = bootstrap rows + aggregate predictions (vote/mean).
- Best with high‑variance base learners; reduces variance while keeping bias low.
- OOB evaluation offers a handy validation signal without a separate holdout.
- Prefer soft voting/averaging for probabilistic outputs; tune sample/feature fractions for diversity.

