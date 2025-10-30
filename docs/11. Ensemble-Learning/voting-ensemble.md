---
sidebar_position: 2
title: Voting Ensemble
---

## Overview

Voting is a simple, powerful ensemble technique that combines predictions from multiple base models (classifiers or regressors) to produce a more accurate and robust final prediction. It embodies the “Wisdom of the Crowd”: aggregate diverse opinions to improve decisions.

## Core idea and methodology

- Train multiple models independently on the same dataset (e.g., Logistic Regression, Naive Bayes, SVM), or multiple instances of the same algorithm with different hyperparameters.
- For a new sample, collect each model’s prediction and aggregate:
	- Classification — majority vote (hard voting) or average of class probabilities (soft voting).
	- Regression — arithmetic mean of predictions.

## Why voting works

- Diversity and error mitigation: models with different biases/inductive biases make different errors; aggregation cancels out individual mistakes.
- Intuition: like a public poll (KBC audience poll, IMDb ratings), many independent judgments average out noise and outliers.
- Simple probability argument (3 independent classifiers with accuracy p):

	$$
	P(\text{ensemble correct}) = p^3 + 3\,p^2(1-p) = 3p^2 - 2p^3.
	$$

	For $p=0.7$, this is $0.343 + 0.441 = 0.784 > 0.7$.

Key assumptions:

1. Base models should be at least better than random (for binary, accuracy > 0.5).
2. Errors should be not perfectly correlated; more diversity → larger ensemble gains.

## Types of voting (classification)

1) Hard voting (majority class labels)

- Each model outputs a class label; the final prediction is the mode of labels.
- Simple and robust when probability calibration is poor.

2) Soft voting (average probabilities)

- Each model outputs class probabilities; the ensemble averages them and picks the class with the highest mean probability.
- Often stronger than hard voting, especially when probabilities are well‑calibrated.

Note: Try both; pick what cross‑validation prefers for your dataset.

## Weights

- You can assign weights to models to reflect their relative strengths (e.g., give more weight to a stronger base learner). In soft voting, weights scale probabilities; in hard voting, they scale votes.

## Practical guidance

- Encourage diversity with different algorithms and/or different hyperparameters.
- For soft voting, prefer models with calibrated probabilities (e.g., Logistic Regression; for others use `CalibratedClassifierCV`).
- Remove clearly weak models (near‑random or < 0.5 accuracy) as they can hurt performance.
- Validate with cross‑validation; tune weights and voting type.

## scikit‑learn: classification

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

log_reg = LogisticRegression(max_iter=1000, random_state=42)
gnb = GaussianNB()
svc = SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42)

voting_clf = VotingClassifier(
		estimators=[
				("lr", log_reg),
				("nb", gnb),
				("svm", svc),
		],
		voting="soft",        # "hard" or "soft"
		weights=[2, 1, 2],     # optional; tune via CV
		n_jobs=-1,
)

voting_clf.fit(X_train, y_train)
print(voting_clf.score(X_test, y_test))
```

Tips:

- If using `voting="soft"`, ensure each estimator supports `predict_proba` (SVC needs `probability=True`).
- To calibrate probabilities (e.g., for SVC without probability=True), wrap with `CalibratedClassifierCV`.

## scikit‑learn: regression

```python
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

reg1 = DecisionTreeRegressor(max_depth=8, random_state=42)
reg2 = Ridge(alpha=1.0, random_state=42)
reg3 = KNeighborsRegressor(n_neighbors=7)

voting_reg = VotingRegressor(
		estimators=[("tree", reg1), ("ridge", reg2), ("knn", reg3)]
)

voting_reg.fit(X_train, y_train)
y_pred = voting_reg.predict(X_test)
```

## Using multiple instances of the same algorithm

Another effective pattern is to vary hyperparameters of a single algorithm to induce diversity:

```python
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

svm1 = SVC(kernel="linear", probability=True, random_state=42)
svm2 = SVC(kernel="poly", degree=2, probability=True, random_state=42)
svm3 = SVC(kernel="rbf", gamma="scale", probability=True, random_state=42)

ensemble = VotingClassifier([
		("svm_lin", svm1),
		("svm_poly", svm2),
		("svm_rbf", svm3),
], voting="soft")
```

## When to use voting

- As a quick, strong baseline ensemble after you’ve built several decent single models.
- When you want easy gains with minimal engineering effort.
- On tabular datasets where combining classic algorithms performs well.

## See also

- [Introduction to Ensemble Learning](./introduction)
- Next up: Stacking, Bagging (Random Forest), and Boosting.

