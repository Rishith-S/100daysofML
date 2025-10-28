---
title: Precision, Recall, F1 — and all confusion‑matrix formulas
sidebar_position: 8
---

Accuracy can be misleading on **imbalanced datasets**. Metrics derived from the **confusion matrix** give a clearer picture by separating kinds of errors. This page defines Precision, Recall, F1, and a complete set of rates you can compute from TP/TN/FP/FN.

## Why not accuracy alone?
- If one class dominates, a trivial model that predicts the majority class can achieve very high accuracy but be useless (e.g., “no terrorist” for everyone).
- Use metrics that focus on the minority/positive class and the types of mistakes: **Precision** (quality of positive predictions) and **Recall** (coverage of actual positives).

## Confusion matrix (binary, rows = actual, cols = predicted)

|               | Predicted 1 | Predicted 0 |
|---------------|-------------|-------------|
| Actual 1      | TP          | FN          |
| Actual 0      | FP          | TN          |

Shorthand:
- TP: True Positive
- TN: True Negative
- FP: False Positive (Type 1 error)
- FN: False Negative (Type 2 error)

Let `P = TP + FN` (actual positives), `N = TN + FP` (actual negatives), and `T = P + N` (total).

## Core metrics
- **Precision (Positive Predictive Value, PPV)**: `TP / (TP + FP)`
	- “Of all predicted positives, how many are truly positive?”
	- Use when the cost of FP is high (e.g., legitimate email marked spam).

- **Recall (Sensitivity, True Positive Rate, TPR)**: `TP / (TP + FN)`
	- “Of all actual positives, how many did we catch?”
	- Use when the cost of FN is high (e.g., missed cancer diagnosis).

- **F1 Score (harmonic mean of Precision and Recall)**: `2 * (Precision * Recall) / (Precision + Recall)`
	- Penalises if either Precision or Recall is low.

- **Fβ Score (tunable trade‑off)**: `(1 + β^2) * (P * R) / (β^2 * P + R)`
	- `β > 1` emphasises Recall; `β < 1` emphasises Precision.

## More confusion‑matrix formulas (the full toolbox)
- **Specificity (True Negative Rate, TNR)**: `TN / (TN + FP)`
- **Fall‑Out (False Positive Rate, FPR)**: `FP / (FP + TN)` = `1 − Specificity`
- **Miss Rate (False Negative Rate, FNR)**: `FN / (FN + TP)` = `1 − Recall`
- **Negative Predictive Value (NPV)**: `TN / (TN + FN)`
- **False Discovery Rate (FDR)**: `FP / (TP + FP)` = `1 − Precision`
- **False Omission Rate (FOR)**: `FN / (FN + TN)` = `1 − NPV`
- **Prevalence** (positive class rate in data): `(TP + FN) / T`
- **Accuracy**: `(TP + TN) / T`
- **Balanced Accuracy**: `(TPR + TNR) / 2`
- **Matthews Correlation Coefficient (MCC)**:
	- `(TP*TN − FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))` (robust single‑number summary on imbalance)

## Binary vs multi‑class
- Binary reports usually focus on the “positive” class (label `1`).
- Multi‑class: compute Precision/Recall/F1 **per class** from the `N × N` confusion matrix.
- Combine per‑class scores via:
	- **Macro average**: unweighted mean over classes; good when classes are balanced or equally important.
	- **Weighted average**: class‑frequency‑weighted mean; useful on imbalanced data.
	- **Micro average**: compute global TP/FP/FN across classes, then derive metrics; dominated by large classes.

## Thresholds and curves
- Classifiers often output scores/probabilities; a threshold (default `0.5`) turns them into labels.
- Increasing the threshold usually increases Precision and decreases Recall (and vice versa).
- Plot **Precision–Recall curves** to study this trade‑off; summarise with **Average Precision (AP)** on imbalanced datasets.

## Minimal scikit‑learn examples
```python
from sklearn.metrics import (
		precision_score, recall_score, f1_score,
		confusion_matrix, precision_recall_fscore_support,
		classification_report
)

y_true = [1, 0, 1, 1, 0, 0, 1]
y_pred = [1, 0, 0, 1, 0, 0, 1]

print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

# Per-class and aggregated
prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, average=None)
print("Per-class precision:", prec)
print("Per-class recall:", rec)
print("Per-class f1:", f1)

print(classification_report(y_true, y_pred, digits=3))  # includes macro/weighted averages
```

## Quick guidance
- Prefer **Precision** when FP is costly; prefer **Recall** when FN is costly.
- Report **F1** when both matter and you need a single score.
- On imbalance, inspect **PR curves**, **support counts**, and consider **macro/weighted** averages.

