---
sidebar_position: 1
title: Decision Trees — Introduction
---

## Decision Trees: An Overview

Decision Trees are a family of models that learn a set of nested if–else questions to classify or predict a target. They are intuitive, require little preprocessing, and mirror human decision-making.

### How Decision Trees Work

- Logic-based splitting: At each node the data is split using a condition on a feature.
	- Categorical features: branch per category (e.g., Outlook ∈ [Sunny, Overcast, Rainy]).
	- Numerical features: choose a threshold (e.g., PetalLength < 2.0) that best separates the labels.
- Tree structure:
	- Root node: the very first question/feature used to split the whole dataset.
	- Decision nodes: internal nodes where further questions are asked.
	- Splitting point: the value/condition used to divide data at a node.
	- Leaves: terminal nodes that output a class or value.
- Recursive process: Repeatedly pick the best split for each subset until a stopping rule is met (pure node, max depth, min samples, etc.).

### Geometric intuition

Each split is an axis-aligned hyperplane that partitions the feature space into rectangles (2D), boxes (3D), or, in general, hyper-cuboids. Traversing the tree amounts to locating which hyper-cuboid a sample falls into.

### Building a Decision Tree (high level)

1. Start with the full dataset (features X and label y).
2. Pick the best feature/threshold at the root using a purity criterion.
3. Split the data into children according to that rule.
4. Recurse on each child until stopping criteria are satisfied.

### Terminology

- Root node — first split of the dataset.
- Splitting point — value/condition used to split at a node.
- Decision node — non-leaf node where a test is performed.
- Leaf node — terminal node that outputs a prediction.
- Branch/Subtree — a node and everything below it.

### Advantages

- Intuitive and easy to visualize/explain.
- Minimal preprocessing (no scaling/normalization required; handles mixed data types).
- Fast inference: prediction follows one path (roughly logarithmic in number of leaves).

### Disadvantages

- Prone to overfitting without depth/min-samples regularization or pruning.
- Can be biased on imbalanced datasets; class weights or sampling may be needed.

### Applications

Widely used for classification and regression (CART). Real-world analogies include games like Akinator that narrow down possibilities via a sequence of yes/no questions.

---

## Entropy: Measuring Disorder

Entropy measures the impurity/uncertainty of a label distribution.

- Definition (base 2):

	$$
	H(P) = -\sum_{c=1}^{C} P(c)\, \log_{2} P(c)
	$$
- Intuition: higher when classes are mixed/equiprobable; zero when all samples belong to one class.
- Two-class examples:
	- 5 Yes / 5 No: H = −(0.5 log₂ 0.5 + 0.5 log₂ 0.5) = 1.
	- 9 Yes / 0 No: H = 0 (perfect purity).
- Key properties (binary case): H ∈ [0, 1]; maximum at P(positive) = 0.5.

### Numerical features with Entropy

To split a numerical column:
1. Sort unique values of the feature.
2. Consider each candidate threshold t between consecutive values.
3. For the two children (x ≤ t and x > t), compute their entropies.
4. Compute the weighted average child entropy.
5. Information Gain is parent entropy minus that weighted average; choose t with maximum gain.

## Information Gain: Choosing the Best Split

Information Gain (IG) quantifies the reduction in entropy after a split.

-
	$$
	IG = H(\text{parent}) - \sum_{k} \frac{\lvert\,\text{child}_{k}\,\rvert}{\lvert\,\text{parent}\,\rvert}\, H(\text{child}_{k})
	$$
- The best attribute/threshold maximizes IG. Recursively applying this selects the split at every node until leaves are pure or stopping rules apply.

---

## Gini Impurity: An Alternative Metric

Gini Impurity measures the probability of mislabeling a random sample if you label it according to the node’s class distribution.

- Definition:

	$$
	G = 1 - \sum_{c=1}^{C} p_c^{2}
	$$
- Examples:
	- 5 Yes / 5 No: G = 1 − (0.5² + 0.5²) = 0.5.
	- 9 Yes / 0 No: G = 0.

### Entropy vs. Gini

- Both are 0 for perfectly pure nodes and peak for evenly mixed classes.
- In binary classification, max H = 1 while max G = 0.5.
- Gini is usually faster to compute (squares vs. logs) and is the default in CART; trees from both criteria are often very similar.

---

## Practical notes

- Control overfitting with constraints: max_depth, min_samples_split/leaf, max_leaf_nodes, pruning.
- Handle imbalance with class weights or resampling.
- For interpretability, limit depth and prefer coarser splits; for accuracy, consider ensembles (Random Forests, Gradient Boosted Trees).

