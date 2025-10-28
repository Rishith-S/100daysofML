---
title: Softmax (Multinomial) Logistic Regression
sidebar_position: 9
---

Softmax Regression generalises Logistic Regression from two classes to **K > 2** classes. It outputs a full probability distribution over classes and is a core building block in deep learning.

## What is it?
- Also called **Multinomial Logistic Regression**.
- For `K = 2`, softmax reduces to ordinary logistic regression (so logistic is a special case of softmax).
- Use cases: any multi‑class classification (e.g., Iris dataset with 3 species).

## The softmax function (probabilities)
Given class scores `z = [z_1, z_2, …, z_K]` (often `z_k = W_k · x + b_k`), softmax produces probabilities `p_k`:

`p_k = exp(z_k) / Σ_j exp(z_j)`

Properties:
- `0 ≤ p_k ≤ 1` for all k
- `Σ_k p_k = 1`

Numerical stability (recommended): subtract the maximum score before exponentiating (log‑sum‑exp trick):

`p_k = exp(z_k − max(z)) / Σ_j exp(z_j − max(z))`

## Intuition (one‑vs‑rest view)
- One helpful mental model: one‑hot encode the labels (e.g., `[1,0,0]`, `[0,1,0]`, …) and imagine training **K logistic regressions**, one per class.
- Each “class‑k logistic” learns its own weights and produces a score; softmax then **normalises all class scores** into a probability distribution.

## Prediction pipeline
1. Compute scores `z_k = W_k · x + b_k` for all classes.
2. Apply softmax to get `p_k` for each class.
3. Predicted label is `argmax_k p_k`.

## Actual training (multinomial cross‑entropy)
The efficient implementation trains a **single model** with all class weights jointly, by minimising **cross‑entropy** over all samples and classes.

Per‑sample loss with one‑hot target `y = [y_1,…,y_K]` and probabilities `p = [p_1,…,p_K]`:

`L = − Σ_k y_k * log(p_k)`

Equivalently, if the true class is `t`, `L = − log(p_t)`.

Useful gradient fact (for optimisation): with `z` as the pre‑softmax scores, `∂L/∂z_k = p_k − y_k`.

Regularisation (L2/L1/Elastic Net) is commonly added, and solvers (LBFGS, SAG/SAGA, etc.) perform gradient‑based optimisation.

## Scikit‑learn example (Iris)
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = make_pipeline(
	StandardScaler(),
	LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42)
)

model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))

# Probabilities for each class (softmax output)
proba = model.predict_proba(X_test[:3])
print("Predict proba (first 3):\n", proba)
print("Predicted labels:", model.predict(X_test[:3]))
```

## Notes and tips
- Standardise features when scales differ; optimisation converges faster.
- For many classes or sparse, high‑dimensional data, `solver="saga"` scales well and supports L1/Elastic Net penalties.
- Use `predict_proba` to inspect the full class distribution; don’t rely only on the argmax label.

## Key takeaways
1. Softmax maps class scores to a valid probability distribution that sums to 1.
2. Training minimises **multinomial cross‑entropy** across all classes simultaneously.
3. Logistic regression is the `K=2` special case of softmax.
4. Numerically stable softmax uses the max‑subtraction trick.
5. Libraries like scikit‑learn provide this via `LogisticRegression(multi_class="multinomial")`.

