---
sidebar_position: 3
title: XGBoost Classification
---
 
## XGBoost Classification: Detailed Notes and Intuition

This page explains how XGBoost performs binary classification, from the objective and gradients to split selection, probability conversion, and practical settings. Multi‑class notes are included at the end.

## Problem setup and base score

- Data: $D = \{(x_i, y_i)\}_{i=1}^n$ with $y_i \in \{0,1\}$
- Predictions are made in log‑odds (logit) space. Start from a constant base score:

$$
b = \log\frac{p}{1-p},\quad p=\frac{1}{n}\sum_i y_i
$$

All initial predictions are $\hat{y}_i^{(0)} = b$. Convert any logit $z$ to probability with the sigmoid $\sigma(z)=\frac{1}{1+e^{-z}}$.

## Objective and derivatives (logistic loss)

Negative log‑likelihood with logits $\hat{y}$:

$$
\ell(y, \hat{y}) = -\, y\,\log\sigma(\hat{y}) - (1-y)\,\log\big(1-\sigma(\hat{y})\big)
$$

Per‑sample first and second derivatives used by XGBoost:

- Gradient: $g_i = \sigma(\hat{y}_i^{(t-1)}) - y_i$
- Hessian: $h_i = \sigma(\hat{y}_i^{(t-1)})\,\big(1-\sigma(\hat{y}_i^{(t-1)})\big)$

## Building one tree (second‑order boosting)

For a candidate tree structure, aggregate over each leaf $j$:

- $G_j = \sum_{i\in I_j} g_i$,  $H_j = \sum_{i\in I_j} h_i$

With regularization $\Omega(f)=\gamma T + \tfrac{1}{2}\lambda\sum w_j^2$ the optimal leaf weight and gain are:

$$
w_j^{\ast} = -\frac{G_j}{H_j + \lambda}
$$

$$
	ext{Gain} = \tfrac{1}{2}\left(\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G_P^2}{H_P+\lambda}\right) - \gamma
$$

Choose the split with the highest positive gain; otherwise prune (controlled by $\gamma$). Note that for classification $h_i$ varies with the current probability, unlike squared‑error regression where $h_i=1$.

## Prediction update and probabilities

After fitting tree $f_t$ the logit prediction updates with learning rate $\eta$:

$$
\hat{y}^{(t)}(x) = \hat{y}^{(t-1)}(x) + \eta\, f_t(x)
$$

Convert to probability $p=\sigma(\hat{y})$. Class labels use a threshold $\tau$ (default $0.5$, often tuned for imbalance). Metrics like ROC‑AUC or PR‑AUC are threshold‑independent and recommended during tuning.

## Handling class imbalance

- Use `scale_pos_weight \approx (\text{negatives}/\text{positives})` for skewed data.
- Prefer AUC/PR‑AUC and tune the probability threshold for F1, recall, or custom cost.
- Row/column sampling (`subsample`, `colsample_bytree`) and regularization further improve generalization.

## Hyperparameters cheat sheet

- Core: `objective='binary:logistic'`, `learning_rate` (η), `n_estimators` with `early_stopping_rounds`
- Tree shape: `max_depth`, `min_child_weight`, `gamma`
- Sampling: `subsample`, `colsample_bytree`
- Regularization: `reg_lambda` (L2), `reg_alpha` (L1)
- Performance: `tree_method='hist'` (or `'gpu_hist'`), `n_jobs=-1`

## Minimal example (scikit‑learn API)

```python
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=8000, n_features=20, weights=[0.7, 0.3], random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=4000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    verbose=False,
    early_stopping_rounds=200,
)

proba = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, proba)
print({"auc": auc, "best_iteration": model.best_iteration})
```

## Diagnostics and interpretation

- Inspect confusion matrix and choose a probability threshold aligned with business costs.
- Use ROC/PR curves to understand trade‑offs; PR‑AUC is more informative under heavy imbalance.
- SHAP values provide faithful local/global explanations for tree models.

## Multi‑class notes

- Set `objective='multi:softprob'` and `num_class=K` to get class probabilities via softmax.
- The same second‑order framework applies; gradients/hessians come from softmax cross‑entropy.

## Common pitfalls

- Relying on accuracy with imbalanced data; prefer AUC/PR‑AUC and calibrated thresholds.
- Too large `learning_rate` with too few trees; use smaller `learning_rate` plus early stopping.
- Forgetting `stratify` during train/valid split, leading to skewed validation sets.
- Mismatch between label encoding and expectation (ensure labels are 0/1 for binary:logistic).