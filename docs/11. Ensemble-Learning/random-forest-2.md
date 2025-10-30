---
sidebar_position: 5
title: Random Forest — Feature Importance, Tuning, and OOB Evaluation
---

## Feature Importance

Feature importance quantifies how much each input column contributes to a model’s predictions.

### Why it matters

- Feature selection: keep the most useful features, remove weak ones to reduce overfitting and training time.
- Interpretability: explain decisions (e.g., why a loan was rejected) by showing which features drove the prediction.

### Which models provide importances

- Tree‑based ensembles: Random Forest, Gradient Boosting, AdaBoost (with trees), Decision Trees.
- Many gradient‑boosting libraries (XGBoost, LightGBM, CatBoost) also expose importances.

### Example intuition (MNIST)

On 28×28 digit images, Random Forest often assigns higher importance to central pixels than to corners, reflecting where information content is concentrated.

### How trees compute impurity‑based importance

At a split node with parent sample set S (size |S|), left child L and right child R, and impurity measure I(·) (e.g., Gini, entropy for classification; MSE for regression), the split’s contribution is the impurity reduction weighted by node size:

$$
\Delta I = \frac{|S|}{N}\,\Big[\, I(S) - \Big( \tfrac{|L|}{|S|} I(L) + \tfrac{|R|}{|S|} I(R) \Big) \Big]
$$

For a given feature k, sum \(\Delta I\) over all nodes that split on k. Normalise across features so importances sum to 1 in a single tree.

### Random Forest importances

Compute importances per tree as above, then average across trees (and re‑normalise). Access via `feature_importances_` after fitting.

### Caveat: high‑cardinality bias

Impurity‑based importances can overvalue features with many unique values. Prefer model‑agnostic checks like permutation importance for validation:

```python
from sklearn.inspection import permutation_importance

rf.fit(X_train, y_train)
result = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
perm_importances = result.importances_mean
```

Tools like SHAP values can provide even deeper local and global explanations.

---

## Hyperparameter Tuning

Random Forests work well out‑of‑the‑box, but tuning can yield further gains.

Commonly tuned parameters

- Number of trees: `n_estimators` (start in 100–500, increase until gains plateau).
- Features per split: `max_features` ("sqrt", "log2", or a fraction like 0.3–0.8).
- Row sampling per tree: `max_samples` (only if `bootstrap=True`; try 0.5–0.75).
- Tree depth/size controls: `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes`.
- Split criterion: classifier ("gini", "entropy"/"log_loss"); regressor ("squared_error", "absolute_error", "poisson").

GridSearchCV (exhaustive)

- Try a small, focused grid on key parameters; works well on smaller datasets.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
param_grid = {
	"n_estimators": [100, 200, 400],
	"max_features": ["sqrt", "log2", 0.5],
	"min_samples_leaf": [1, 2, 5],
}
grid = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```

RandomizedSearchCV (faster on large spaces)

- Sample a fixed number of parameter combinations; often finds near‑optimal settings quickly.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
param_dist = {
	"n_estimators": randint(100, 600),
	"max_features": uniform(0.3, 0.7),
	"min_samples_leaf": randint(1, 10),
}
search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=40, cv=5, n_jobs=-1, random_state=42)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
```

Notes

- Use stratified CV and class‑weighted metrics for imbalanced classification.
- Calibrate expectations: smaller `max_features` increases diversity (reduces correlation) but can raise bias — tune with CV.

---

## Out‑of‑Bag (OOB) Evaluation

With bootstrapping, each tree trains on a sample drawn with replacement from the training set. Roughly 36.8% of rows are left out (OOB) for a given tree, since the probability of not being picked is \((1 - 1/N)^N \approx e^{-1}\).

These OOB rows serve as an internal validation set:

- Enable with `oob_score=True` (requires `bootstrap=True`).
- After fitting, read `oob_score_` for a quick performance estimate (accuracy for classification; R² for regression).

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300, oob_score=True, bootstrap=True, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
print("OOB score:", rf.oob_score_)
```

OOB scores are typically close to test‑set metrics but may differ slightly; still, they are very handy when you want validation without a dedicated holdout.

