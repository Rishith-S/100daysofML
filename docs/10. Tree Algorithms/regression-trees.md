---
sidebar_position: 3
title: Regression Trees
---

## What are Regression Trees?

Regression Trees are Decision Trees used for regression problems where the target is continuous (numerical). Unlike Classification Trees that output a class label, Regression Trees output a numeric value.

They learn a sequence of if–else splits on features and make a piece‑wise constant prediction: each leaf stores a single value (typically the mean of training targets in that leaf).

## Why use Regression Trees over Linear Regression?

- Linear Regression assumes a roughly linear relationship between features and the target.
- When relationships are non‑linear, piece‑wise models like trees can fit the structure better.
- Example: study hours vs. exam marks may rise, plateau, then drop at very high hours. A straight line fits poorly; a tree can carve the space into ranges and assign appropriate values per range.

## How Regression Trees split the data

At each node, the algorithm considers splits of the form “feature j ≤ threshold t” that divide the data into left/right groups, then picks the split that minimises prediction error in the children.

- Leaf prediction (mean target in the node):

	$$
	\hat{y}_{S} = \frac{1}{|S|} \sum_{i\in S} y_i
	$$

- Impurity at a node S (common choices):
	- Mean Squared Error (MSE):

		$$
			ext{MSE}(S) = \frac{1}{|S|} \sum_{i\in S} (y_i - \hat{y}_S)^2
		$$

	- Mean Absolute Error (MAE):

		$$
			ext{MAE}(S) = \frac{1}{|S|} \sum_{i\in S} |y_i - \hat{y}_S|
		$$

- Split selection: for a candidate threshold t that partitions parent set P into children L and R, minimise the weighted child impurity:

	$$
	\mathcal{J}(t) = \frac{|L|}{|P|} \cdot \text{Impurity}(L) + \frac{|R|}{|P|} \cdot \text{Impurity}(R)
	$$

Repeat this recursively until stopping criteria are met (e.g., max depth, min samples, or no split improves impurity enough).

### Concrete intuition

If “hours < 3” typically yields low marks, and “3 ≤ hours ≤ 6” yields higher marks, the tree will split at 3 (and maybe again at 6). Each resulting leaf predicts the average marks of samples in that interval.

## Key hyperparameters (tuning knobs)

- Criterion (split quality)
	- mse (squared error) or mae (absolute error). In scikit‑learn you’ll see names like "squared_error" (formerly "mse") and "absolute_error" (formerly "mae").
- Splitter
	- best: try all candidates; can overfit.
	- random: random subset of candidates; adds regularising noise.
- max_depth
	- Maximum tree depth. Low → underfit; high → overfit.
- min_samples_split
	- Minimum samples required to split a node. Higher values simplify the tree.
- min_samples_leaf
	- Minimum samples in any leaf after a split. Strong regulariser against tiny, unstable leaves.
- max_leaf_nodes
	- Upper bound on the number of leaves; lower means simpler trees.
- min_impurity_decrease
	- Require a minimum reduction in impurity to split; filters weak splits.
- max_features
	- Number (or fraction) of features to consider at each split. Useful in high‑dimensional data and in ensembles to reduce overfitting.

See also: [Overfitting, Underfitting and Hyperparameters](./hyperparameters) for deeper guidance and typical value ranges.

## Feature importance

Trees compute feature importances from the total impurity reduction they attribute to each feature across the tree. After training, inspect `feature_importances_` to understand which columns contributed most and for feature selection.

## Hyperparameter search

Use cross‑validated search to find a good combination:

- Grid Search CV: systematic search over a parameter grid.
- Randomized Search CV: samples parameter combinations randomly; faster on large spaces.

## Practical recommendations

- Single trees are easy to interpret but can overfit. Prefer ensembles for stronger performance and robustness:
	- Random Forests (bagging of trees)
	- Gradient Boosted Trees / XGBoost / LightGBM (boosting)
- Still, a well‑regularised single tree is a great baseline and is very fast at inference.

## Minimal example (scikit‑learn)

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

# X: numpy array or pandas DataFrame of shape (n_samples, n_features)
# y: 1D array of target values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeRegressor(
		criterion="squared_error",  # or "absolute_error"
		max_depth=None,
		min_samples_split=2,
		min_samples_leaf=1,
		splitter="best",
		random_state=42,
)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse:.3f}")

# Optional: quick hyperparameter search
param_dist = {
		"max_depth": [3, 5, 8, None],
		"min_samples_leaf": [1, 5, 10, 20],
		"min_samples_split": [2, 10, 20, 50],
		"splitter": ["best", "random"],
}

search = RandomizedSearchCV(tree, param_distributions=param_dist, n_iter=20, cv=5,
														scoring="neg_root_mean_squared_error", random_state=42)
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
```

## Quick takeaways

- Trees make piece‑wise constant predictions by recursive binary splits.
- Use MSE/MAE as the impurity criterion for regression.
- Regularise with max_depth, min_samples_leaf/split, max_leaf_nodes, and min_impurity_decrease.
- Prefer ensembles for top accuracy; use a single tree as a simple, interpretable baseline.

