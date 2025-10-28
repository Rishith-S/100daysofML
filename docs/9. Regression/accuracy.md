---
title: Accuracy and the Confusion Matrix
sidebar_position: 7
---

This page introduces core classification metrics, focuses on **accuracy**, explains its limits, and shows how the **confusion matrix** exposes the kinds of mistakes a model makes.

## Classification metrics overview
- Metrics quantify how well a classifier performs, similar to how regression metrics evaluate regressors.
- Common metrics: accuracy, confusion matrix, precision, recall, F1‑score, and AUC‑ROC.

## Accuracy
- Definition: `Accuracy = (# correct predictions) / (# total predictions)`.
- Use: A simple absolute indicator and a way to compare models on the same test set.
- Examples:
	- Binary: if 8 of 10 predictions are correct → accuracy = `0.8` (80%).
	- Multi‑class: same calculation—count every exact match between predicted and actual label.
- What is a “good” accuracy? It’s **problem dependent**:
	- High‑stakes domains (medical diagnosis, self‑driving): even 99% may be unacceptable.
	- Low‑stakes domains (marketing clicks, weekend orders): 80% may be fine.
- Limitation: One number that hides **what kinds of errors** occurred; it does not tell you whether the errors were more positive→negative or negative→positive.

## Confusion matrix
The confusion matrix reveals the **type of mistakes** by comparing predictions against ground truth.

Convention used here: columns = predicted, rows = actual.

|               | Predicted 1 | Predicted 0 |
|---------------|-------------|-------------|
| Actual 1      | TP          | FN          |
| Actual 0      | FP          | TN          |

- TP (True Positive): predicted 1 and actual 1.
- TN (True Negative): predicted 0 and actual 0.
- FP (False Positive, Type 1 error): predicted 1 but actual 0.
- FN (False Negative, Type 2 error): predicted 0 but actual 1.

Related formulas:
- `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
- `Precision (Positive class) = TP / (TP + FP)`
- `Recall (Sensitivity, TPR) = TP / (TP + FN)`
- `F1 = 2 * Precision * Recall / (Precision + Recall)`

### Multi‑class extension
For `N` classes, the confusion matrix becomes `N × N`. The **diagonal** cells count correct predictions per class; off‑diagonal cells show specific misclassifications (e.g., how often class 2 is confused as class 0).

## Type 1 and Type 2 errors
- Type 1 (False Positive): predict positive when actually negative.
- Type 2 (False Negative): predict negative when actually positive.

## Why accuracy can mislead on imbalanced data
When one class dominates (e.g., 99,999 normal vs 1 positive), a trivial classifier that always predicts the majority class achieves ~100% accuracy yet is **useless**. In such settings, prefer **precision/recall**, **F1**, or **ROC/PR curves**.

## Minimal scikit‑learn example
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_true = [1, 0, 1, 1, 0, 0, 1]
y_pred = [1, 0, 0, 1, 0, 0, 1]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=3))  # precision/recall/F1 per class
```

## Key takeaways
1. Accuracy is simple and useful for quick comparisons, but it hides **which** errors occur.
2. The confusion matrix decomposes errors into TP/TN/FP/FN and generalises to multi‑class.
3. On imbalanced datasets, accuracy can be dangerously misleading—use precision/recall/F1 or PR/ROC analysis.

