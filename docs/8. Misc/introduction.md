---
sidebar_position: 1
---

# Bias vs Variance in Machine Learning

---

## 📌 **Bias**

### 🔍 Definition

Bias refers to the **inability of a machine learning model to accurately capture the true relationship underlying the data**.

It measures how much the predicted values differ from the actual values.

---

### 🔺 High Bias

- The model **performs poorly on training data**.
- Indicates that the model is **not complex enough** to understand the patterns.
- Suggests the model is **underfitting**.
- 🧠 **Analogy**: A student who doesn't study enough and thus performs poorly even on familiar questions.
- 📈 **Visual**: A straight line trying to fit a curved dataset — shows large errors even in training.

---

### 🔻 Low Bias

- The model **performs well on training data**.
- It has **effectively captured the relationship** within the training set.

---

## 🔄 **Variance**

### 🔍 Definition

Variance refers to the **sensitivity of a machine learning model to small fluctuations in the training data**.

It measures how much the model's performance changes with different training sets.

---

### 🔺 High Variance

- The model **performs well on training data but poorly on test data**.
- Indicates the model has **memorized noise** — fails to generalise.
- Suggests the model is **overfitting**.
- 🧠 **Analogy**: A student who memorizes exact answers and fails when questions are reworded.
- 📈 **Visual**: A highly complex curve that fits every training point but performs badly on new data.

---

### 🔻 Low Variance

- The model has **consistent performance across datasets**.
- It is **not overly sensitive** to changes in training data.
- **Can generalise well** to unseen data.

---

## ⚖️ **Relationship to Overfitting and Underfitting**

### 🚫 Underfitting

- The model **fails to capture patterns** in training data.
- Characterized by:
    - **High Bias**
    - **Low Variance**
- Performs poorly on both training and test data.

---

### ⚠️ Overfitting

- The model **memorizes training data too closely**.
- Characterized by:
    - **Low Bias**
    - **High Variance**
- Performs well on training data but poorly on test data.

---

## ⚖️ **Bias-Variance Trade-off**

- A **core concept** in machine learning.
- Goal: Find a model with **both low bias and low variance**.
- Challenge:
    - Reducing bias ⬆️ often increases variance ⬆️
    - Reducing variance ⬇️ often increases bias ⬆️
- 🔍 This is the **bias-variance trade-off**.

---

### ✅ Ideal Scenario

- A model that strikes the **right balance**: moderate bias and variance.
- **Generalises well** without overfitting or underfitting.

---

### 🧰 Techniques to Handle the Trade-off

- **Regularisation**:
    - **L1 (Lasso)** and **L2 (Ridge)** regularisation.
    - These simplify the model to reduce variance while managing bias.
    - Help prevent **overfitting** by **penalizing complexity**.
- **Bagging**
- **Boosting**
