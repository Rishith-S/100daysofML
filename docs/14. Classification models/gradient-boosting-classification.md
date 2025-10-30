---
sidebar_position: 3
title: Gradient Boosting Classification
---

## What is Gradient Boosting?

- Gradient Boosting is a crucial machine learning algorithm.
- It is a **boosting technique** that combines multiple small models in a stage-wise fashion.
- The core concept it operates on is **additive modelling**. This involves:
    - Training a first model on the data.
    - The first model makes errors, which are then passed on to the next model.
    - The subsequent model attempts to correct these errors.
    - The combination of these models performs better than the initial one.
    - This process continues, adding models in stages, with each new model learning from the errors of the combined previous models.

## Why does Gradient Boosting work? (Addressing Bias-Variance Trade-off)

- The goal in machine learning is to achieve a low-bias, low-variance model, which is often difficult due to their inverse proportionality.
- Gradient Boosting uses **weak learners** (models with high bias and thus low variance).
- By combining multiple high-bias algorithms through additive modelling, it converts them into a low-bias model. This is described as the "magic of Gradient Boosting".

## Gradient Boosting for Classification vs. Regression

- The beauty of Gradient Boosting is that it does **not require a different algorithm** for classification settings compared to regression.
- The exact same steps can be applied.
- The **only difference** lies in the **loss function** used.
    - For regression, **Mean Squared Error (MSE)** is typically used.
    - For classification, **Log Loss** is used as the loss function. This small change allows the algorithm to solve classification problems.

## Step-by-Step Process for Classification (Practical Example)

1. **Goal**: Build multiple small models and add them in a stage-wise fashion to predict placement (binary classification).
2. **Stage 1: Initial Model (F0(x))**:
    - Always starts with a very simple model.
    - Unlike regression (where the mean is used), in classification, the **log of odds** is used.
    - Log of odds is calculated as `log(number of ones / number of zeros)`. The base `e` (natural logarithm) should be used, not `log base 10`.
    - This log of odds value is the **initial prediction**.
    - **Conversion to Probability**: Since the initial prediction is in log odds, it needs to be converted to a probability to calculate residuals. The formula `e^x / (1 + e^x)` is used for this conversion.
    - This initial model gives a constant probability prediction for all inputs (e.g., always predicting placement based on the overall data distribution). It serves as a good starting point despite being obviously wrong.
3. **Calculating Errors (Pseudo-Residuals)**:
    - Errors are measured as **pseudo-residuals**, which are calculated by subtracting the prediction from the true output (`True Output - Prediction`).
4. **Stage 2: Training the Second Model (F1(x))**:
    - A **new model (a Decision Tree)** is trained to learn from the errors (pseudo-residuals) of the first model.
    - It's important to note that even though it's a classification problem, a **regression tree** is used because the output it's predicting (residuals) is continuous.
    - These decision trees should be **weak learners** (e.g., with a limited number of leaf nodes).
5. **Combining Models and Calculating Output**:
    - The overall output of the combined model is not a simple sum of the log odds directly from the first model and the second model's output.
    - The output of the decision tree (F1(x)) represents **differences of probabilities**.
    - For each leaf node of the decision tree, a **gamma value (log of odds)** is calculated using a specific formula: `sum(residuals) / sum(previous_probabilities * (1 - previous_probabilities))`. This calculation helps convert the decision tree's output into log odds that can be combined.
    - The combined model's log of odds for each data point is then calculated by adding the initial model's log of odds to the decision tree's calculated gamma value for that point.
    - This combined log of odds is again converted back to **probability** using `e^x / (1 + e^x)`.
    - This process shows an **improvement** in prediction accuracy compared to the initial model.
6. **Stage 3 (and subsequent stages): Adding More Models**:
    - The error (pseudo-residuals) of the combined Stage 2 model is calculated.
    - Another regression decision tree (F2(x)) is trained on these new residuals.
    - The gamma values for this new tree's leaf nodes are calculated using the same formula.
    - The overall combined model's log of odds is then `F0(x) + F1(x)_output + F2(x)_output`.
    - This again is converted to probability.
    - As more models are added, the residuals tend to decrease towards zero, indicating improved model accuracy.
7. **Learning Rate**:
    - A **learning rate** (typically 0.1) can be introduced to make the jumps in error reduction more gradual.
    - It multiplies the output (gamma value) of each decision tree before it's added to the combined model, preventing too sharp a change in predictions.

## Predicting for a New Query Point

- To predict for a new data point, it is passed through each of the trained models sequentially.
- The outputs (log odds or gamma values) from all models are summed up.
- The final combined log of odds is then converted to a probability, which gives the classification output (e.g., if probability > 0.5, classify as 1, otherwise 0).

## Geometric Intuition

- To visualise, 2D input data (x1, x2) is often transformed into a 3D space, where the third axis represents the target variable (y, which is 0 or 1). This creates two distinct levels for the data points.
- **Stage 1**: The initial model (F0(x)) plots a **flat decision boundary** (e.g., a plane at y=0.55), which initially classifies all points as one class.
- **Subsequent Stages**: Each additional decision tree attempts to correct the errors (residuals) of the previous combined model.
    - These decision trees learn to predict positive or negative residuals based on different regions of the input space.
    - When combined, these trees progressively **bend and refine the decision boundary**.
    - Initially, the boundary might be a simple plane; with more trees, it becomes more flexible and complex, accurately separating the classes by cutting through the 3D data points.
    - The visual representation shows the decision boundary adapting to the data, eventually separating the two classes effectively.

## Recommendations from the Source

- It is highly recommended to watch the Gradient Boosting for Regression video on the channel first, as it provides foundational understanding for classification.
- An external blog with code is suggested for gaining a deeper geometric intuition and for hands-on experimentation.