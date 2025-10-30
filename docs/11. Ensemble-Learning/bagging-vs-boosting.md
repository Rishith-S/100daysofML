---
sidebar_position: 7
title: Bagging vs Boosting — Key Differences and When to Use
---

Bagging (Bootstrap Aggregation) and Boosting are two fundamental ensemble strategies that improve model performance in different ways. This page distills their core differences, intuition, and practical guidance.

## Three core distinctions

### 1) Type of base models (bias–variance profile)

- Bagging
	- Favors low‑bias, high‑variance learners (e.g., fully grown Decision Trees, KNN, sometimes SVM).
	- Goal: reduce variance by averaging many diverse models trained on resampled data.
- Boosting
	- Starts with high‑bias, low‑variance learners (e.g., very shallow trees/stumps: max_depth ∈ {1,2}).
	- Goal: reduce bias by adding learners sequentially that focus on the previous errors.

### 2) Learning scheme

- Bagging — parallel learning
	- Train many base models independently on bootstrap (or pasted) samples.
	- Easy to parallelise; randomness comes from row/feature sampling.
- Boosting — sequential learning
	- Train models one after another; each model depends on the previous (e.g., reweighted data in AdaBoost or residuals in Gradient Boosting).
	- Hard to parallelise due to stage‑wise dependency.

### 3) Weighting of base learners

- Bagging
	- All learners have equal say in the final prediction (simple vote/mean), though some implementations can use weights post‑hoc.
- Boosting
	- Learners have different weights; better learners get higher weight (e.g., AdaBoost’s α depends on weighted error; gradient boosting combines learners with learning rate/shrinkage).

## Intuition in pictures (words)

- Bagging smooths overfitted, jagged boundaries by averaging many strong-but-unstable models → lower variance, similar bias.
- Boosting refines a simple boundary by iteratively correcting mistakes → lower bias, with regularisation to control variance.

## Typical algorithms

- Bagging family: BaggingClassifier/Regressor, Random Forest (trees + node‑level feature subsampling).
- Boosting family: AdaBoost (SAMME/SAMME.R), Gradient Boosting, XGBoost, LightGBM, CatBoost.

## When to prefer each

- Choose Bagging when
	- Base model is high‑variance (deep trees, KNN, some SVMs).
	- You want parallel training and strong baselines with minimal tuning.
	- Interpretability of single trees plus stability of the ensemble is desired (e.g., Random Forest feature importances).

- Choose Boosting when
	- Base model underfits (high bias) and you need to increase capacity via sequential refinement.
	- You can budget more tuning time (learning rate, number of estimators, tree depth) and accept sequential training.
	- You need top accuracy on tabular data (XGBoost/LightGBM/CatBoost often excel).

## Pros and cons at a glance

| Aspect | Bagging | Boosting |
|-------|---------|----------|
| Training | Parallel | Sequential |
| Focus | Reduce variance | Reduce bias |
| Base learner | High‑variance (deep trees) | Weak learners (stumps/shallow trees) |
| Final weighting | Usually equal vote/mean | Weighted by stage quality (α or learning rate) |
| Robust to noise | High (averaging) | Can be sensitive; regularise carefully |
| Tuning effort | Lower | Higher (lr, estimators, depth, regularisation) |

## Minimal scikit‑learn sketches

Bagging (Random Forest example)

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
```

Boosting (AdaBoost example)

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
		estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
		n_estimators=200,
		learning_rate=0.1,
		algorithm="SAMME.R",
		random_state=42,
)
ada.fit(X_train, y_train)
```

## Quick takeaways (interview‑ready)

- Bagging: parallel, equal votes, best with high‑variance learners; Random Forest is the canonical example.
- Boosting: sequential, weighted votes, best for reducing bias; AdaBoost/Gradient Boosting/XGBoost families are typical.
- Both aim for the holy grail — low bias and low variance — but attack the problem from opposite directions.

