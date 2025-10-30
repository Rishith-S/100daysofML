---
sidebar_position: 2
title: Decision Trees — Overfitting, Underfitting and Hyperparameters
---

Decision Trees are powerful and intuitive, but they can easily overfit. This page explains what overfitting/underfitting look like for trees and how to tune the key hyperparameters in scikit‑learn.

## Overfitting in Decision Trees

- Definition: the model performs extremely well on training data but poorly on unseen data. The tree memorises idiosyncrasies of the training set instead of learning general patterns.
- Mechanism:
  - Without limits, a tree keeps splitting until leaves are almost pure (sometimes only a couple of samples).
  - Predictions for new points then depend on tiny, possibly noisy regions.
  - Large `max_depth` (or unlimited) encourages this behaviour.
- Geometric intuition: to fit training data perfectly, the tree draws many axis‑parallel boundaries, carving tiny rectangles that classify training points perfectly but generalise poorly.

## Underfitting in Decision Trees

- Definition: the model is too simple to capture structure in the data; both train and test scores are low.
- Mechanism:
  - Very shallow trees (e.g., `max_depth = 1`) allow only a few splits, creating broad regions that mix classes.

## Hyperparameters to Control Bias/Variance

Below are the most relevant scikit‑learn parameters and how they affect bias (underfitting) and variance (overfitting).

- `max_depth`
  - Max number of levels in the tree.
  - Higher → lower bias, higher variance (can overfit).
  - Lower → higher bias, lower variance (can underfit).
  - Rule‑of‑thumb: tune with CV; start around 3–12 for tabular tasks.

- `criterion`
  - Split quality: `"gini"` (default, fast) or `"entropy"` (information gain).
  - Both usually yield similar trees; Gini is slightly faster.

- `splitter`
  - Strategy to pick splits at each node.
  - `"best"`: tries all candidates and picks the best (can overfit).
  - `"random"`: random candidates; adds noise that may reduce overfitting.

- `min_samples_split`
  - Minimum samples required to split an internal node.
  - Higher → coarser trees (more bias, less variance).
  - Typical values: 10, 20, 50 (or a float fraction like 0.05).

- `min_samples_leaf`
  - Minimum samples required at a leaf.
  - Strong regulariser: prevents tiny, unstable leaves.
  - Typical values: 1–10 for small sets; 20–100+ for large datasets.

- `max_features`
  - Number of features considered at each split.
  - Using fewer features (e.g., `"sqrt"` or a fraction) injects randomness that can reduce overfitting; this is critical in Random Forests.

- `max_leaf_nodes`
  - Upper bound on the number of leaves; smaller values regularise.

- `min_impurity_decrease`
  - Require a minimum impurity reduction for a split.
  - Larger thresholds prune weak splits early.

## Practical tuning workflow

1. Split your data (train/validation or K‑fold CV).
2. Start with a shallow tree and gradually increase complexity:
   - Tune `max_depth`, `min_samples_leaf`, `min_samples_split` first.
   - Optionally try `splitter="random"` and different `criterion`.
3. Monitor both train and validation metrics to avoid over/underfitting.
4. Prefer reproducibility with `random_state`.
5. When you need stronger performance and robustness, move to ensembles (Random Forests, Gradient Boosted Trees), reusing the same regularisation ideas.

## Example (scikit‑learn)

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    criterion="gini",           # or "entropy"
    max_depth=8,                 # tune via CV
    min_samples_leaf=10,         # prevents tiny leaves
    min_samples_split=20,        # coarser splits
    splitter="best",            # or "random" to reduce variance
    max_features=None,           # try "sqrt" / 0.5 on high‑dim data
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    random_state=42,
)
clf.fit(X_train, y_train)
```

## Key takeaways

- Trees overfit by growing deep and creating tiny leaves; limit depth and leaf sizes.
- Strong, simple controls: `max_depth`, `min_samples_leaf`, `min_samples_split`.
- Evaluate with cross‑validation and tune for the best generalisation, not perfect training accuracy.
