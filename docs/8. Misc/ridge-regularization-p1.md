---
sidebar_position: 2
---

# Ridge Regularization (L2 Regularization) - Part 1

---

## 🎯 **What is Regularization?**

### 🔍 Definition

Regularization is a technique used in machine learning to **induce added information into a model to reduce overfitting**.

- It is a very important concept, especially when dealing with **linear regression** and **logistic regression**.
- It falls under the category of **ensemble techniques**, alongside **bagging** and **boosting**.

---

## ⚠️ **Understanding Overfitting**

### 📊 The Problem

Overfitting occurs when a machine learning model:
- **Performs exceptionally well on the training data** but **performs poorly on the testing data**.
- Gives **completely different results** when applied to a different dataset.

### 🔴 Classic Linear Regression Example

In linear regression, a classic example of overfitting is when the **"best fit line" becomes too steep**:
- May have an **infinite slope** (`m`) if it perfectly passes through only two points in 2D space.
- Results in **poor performance on new data** (high variance, low bias).

---

## 📋 **Types of Regularization**

There are three main types of regularization techniques:

1. **Ridge Regression (L2 Regularization)** ← *Focus of these notes*
2. **Lasso Regression (L1 Regularization)**
3. **Elastic Net** - A combination of Ridge and Lasso

---

## 🔧 **Ridge Regression (L2 Regularization)**

### 💡 Core Idea

Ridge Regression modifies the **standard loss function** (e.g., Mean Squared Error or MSE) of linear regression by **adding a penalty term** that discourages large coefficient values.

---

### 📐 **Loss Function Modification**

#### Original Linear Regression Loss Function

The original linear regression loss function (e.g., MSE) aims to **minimize the difference between actual and predicted values**.

#### Ridge Regression Loss Function

Ridge Regression adds a penalty term to the loss function to reduce overfitting.

Where:
- λ (lambda) = regularization parameter (also called alpha in scikit-learn)
- w_j = coefficients/weights of the model
- p = number of features

**Key Point**: The penalty term penalizes large coefficient values.

---

### ⚙️ **The Lambda (λ) or Alpha (α) Parameter**

The `lambda` parameter (called `alpha` in scikit-learn) is a **hyperparameter** that controls the strength of the penalty.

#### 🔴 **If λ = 0 (Red Line)**

- Ridge Regression behaves **exactly like ordinary linear regression**.
- **No penalty term** is applied.
- Model can become too complex, leading to **overfitting**.
- Characteristics:
    - ✗ High variance
    - ✗ Low bias
    - ✗ Overfitting risk

#### 🟢 **If λ = 20 (Green Line) - Optimal**

- There is an **"optimal" λ value** that provides a **good balance**.
- Model avoids **both overfitting and underfitting**.
- Results in a model that **generalises well** to unseen data.
- Characteristics:
    - ✓ Moderate variance
    - ✓ Moderate bias
    - ✓ Good generalization

#### 🔵 **If λ is Very High (Blue Line)**

- Model is **heavily penalised** for large coefficient values.
- Leads to **simpler models** with very small coefficients.
- Can result in **underfitting** - fails to capture underlying patterns.
- Characteristics:
    - ✓ Low variance
    - ✗ High bias
    - ✗ Underfitting risk

---

### ⚡ **How Ridge Regression Works**

#### Mechanism

1. The penalty term ensures that **coefficient values do not become excessively large**.
2. By penalizing large coefficients, Ridge Regression **discourages overly complex models** that might fit the training data too perfectly (overfitting).
3. The model **balances** between fitting the training data well and keeping coefficients small.

#### Practical Example

Suppose we have two models with similar original loss values:
- **Model A**: Coefficients = [2.25, 1.50, 0.80]
- **Model B**: Coefficients = [2.03, 1.40, 0.75]

With Ridge Regression (λ > 0), **Model B is preferred** because:
- Its penalty term is smaller
- It has simpler, smaller coefficient values
- It is less likely to overfit

---

### 📊 **Impact of Alpha (λ) Visualization**

The regularization impact shows three different scenarios:

| Alpha Value | Behavior | Impact |
|-------------|----------|--------|
| **α = 0** | Fits every training point | Overfitting |
| **α = 20** | Smooth curve, balanced fit | Optimal |
| **α = 200** | Over-simplified, underfits | Underfitting |

When alpha is 0, the model follows every data point including noise (overfitting). As alpha increases, the model becomes smoother and more generalizable. However, if alpha is too high, the model becomes too simple and fails to capture important patterns (underfitting).

---

## 💻 **Practical Implementation with Scikit-learn**

### Dataset

The **diabetes dataset** is used:
- **Input columns**: 10 features (age, sex, BMI, blood pressure, etc.)
- **Target column**: Quantitative target variable (disease progression)

### Key Parameters

```python
from sklearn.linear_model import Ridge

# Default alpha value is 1.0
ridge_model = Ridge(alpha=1.0)
```

### Hyperparameter Tuning

The `alpha` value is **crucial** for Ridge Regression performance:
- **Too low** (close to 0): Similar to linear regression, may overfit
- **Too high** (e.g., 200): Heavy penalty, may underfit
- **Optimal** (e.g., 1.0 - 20): Best generalization

### Performance Comparison

| Model | Test R² Score | Behavior |
|-------|---------------|----------|
| Linear Regression | 0.518 | Baseline |
| Ridge (α=1.0) | ~0.520 | Slightly less complex |
| Ridge (α=20) | ~0.525 | Better generalization |
| Ridge (α=200) | ~0.480 | Underfitting |

---

### 📝 **Ridge Regression Formula Summary**

#### Loss Function

The Ridge Regression loss function combines prediction error with coefficient penalty. The general form adds a regularization term to prevent large coefficients.

#### Solution Formula

The optimal weight coefficients are calculated as:

**w = (X^T X + λI)^(-1) X^T y**

Where:
- X = feature matrix (m×n)
- y = target vector
- w = weight coefficients
- λI = regularization term (I is identity matrix)

---

## 🎯 **Key Takeaways**

✅ **Ridge Regression** adds a penalty term to the loss function to reduce overfitting.

✅ **Lambda (α)** controls the regularization strength:
- λ = 0 → Linear Regression (potential overfitting)
- λ = optimal → Best generalization
- λ = very high → Underfitting

✅ **Bias-Variance Trade-off**: Ridge helps achieve the right balance.

✅ **Coefficients**: Ridge shrinks coefficients proportionally (never to exactly zero).

✅ **Use Cases**: Ridge is particularly useful when you have:
- Multicollinearity (correlated features)
- Many features relative to samples
- Need for model simplicity and generalization
