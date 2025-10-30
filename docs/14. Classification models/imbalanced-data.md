---
sidebar_position: 4
title: Imbalanced Data
---

## What is Imbalanced Data?

- Imbalanced data occurs in machine learning when the classes in a dataset are not represented equally.
- There's a **majority class** with a significantly higher number of instances and a **minority class** with very few instances.
- **Example:** In a college placement prediction problem with 500 students, if 450 students are placed (majority class) and only 50 are not placed (minority class), this is an example of imbalanced data.
- This problem can occur in **binary classification** (two classes) and **multi-class classification** (more than two classes, where one class dominates others).

## Problems Caused by Imbalanced Data
Imbalanced data leads to two major problems:

1. **Bias towards the Majority Class:**
    - Machine learning models learn patterns from the data they are given.
    - With imbalanced data, algorithms see much more data from the majority class, causing them to focus more on it.
    - As a result, the trained model becomes **biased towards the majority class**, performing well on it but very poorly on the minority class.
    - **Example:** A logistic regression model trained on imbalanced placement data might have a decision boundary that strongly focuses on correctly classifying placed students (majority), while largely ignoring unplaced students (minority).
2. **Unreliable Metrics (e.g., Accuracy):**
    - While **accuracy** is often considered the most reliable metric for classification problems, it becomes **unreliable with imbalanced data**.
    - **Example:** For a dataset with 900 placed students and 100 unplaced students, a "dummy model" that simply predicts "placed" for everyone would achieve 90% accuracy. However, this model completely fails to predict unplaced students, making the 90% accuracy misleading.
    - Other metrics like **precision, recall, and ROC AUC score** provide a more realistic picture of the model's performance on minority classes. For instance, a model with 96% accuracy on imbalanced data might have only a 29% recall for the minority class, indicating very poor performance on that critical group.

## Why is Handling Imbalanced Data Important?
Two main reasons highlight the importance of understanding and addressing imbalanced data:

1. **Omnipresent in Industry:**
    - Imbalanced data problems are common in real-world machine learning applications across various industries.
    - **Examples include:**
        - **Finance:** Fraud detection (normal transactions vs. fraudulent ones), credit risk assessment (loan repayments vs. defaults).
        - **Healthcare:** Rare disease prediction (most people don't have the disease, few do).
        - **Manufacturing:** Predicting machine failures (most machines work correctly, few fail).
        - **Consumer Internet Companies:** Churn prediction (most customers stay, few leave).
        - **Geospatial Science:** Earthquake/volcano prediction (rare events).
    - In many of these problems, accurately predicting the **minority class is critically important** (e.g., finding fraudulent transactions, identifying patients with rare diseases).
    - Therefore, data science professionals are highly likely to encounter and work on imbalanced data problems.
2. **Interview Point of View:**
    - It is a frequently asked topic in data science interviews because interviewers want to assess a candidate's practical understanding of real-world challenges beyond just algorithms.

## Techniques for Handling Imbalanced Data
Five important techniques are discussed:

1. **Undersampling:**
    - **Concept:** Reduces the number of data points in the **majority class** to balance it with the minority class.
    - **Mechanism (Random Undersampling):** Randomly removes instances from the majority class until its count matches that of the minority class.
    - **Advantages:**
        - **Reduces bias** towards the majority class, enabling the model to give equal importance to both classes.
        - Can lead to **faster training** if the original dataset was very large, as the resulting dataset is smaller.
    - **Disadvantages:**
        - **Loss of potentially important data**, as a significant portion of the majority class is discarded.
        - **Sampling bias** can be introduced if the random sampling inadvertently ignores certain types of points from the majority class.
    - **Implementation:** The `imbalanced-learn` library provides `RandomUnderSampler`.
2. **Oversampling (Random Oversampling):**
    - **Concept:** Increases the number of data points in the **minority class** to balance it with the majority class.
    - **Mechanism (Random Oversampling):** Duplicates existing data points from the minority class until its count matches that of the majority class.
    - **Advantages:**
        - **Reduces bias** towards the majority class, allowing the model to focus on both classes equally.
    - **Disadvantages:**
        - **Increases dataset size**, potentially doubling it, which can be problematic for very large datasets.
        - **Risk of overfitting**, especially with algorithms like Decision Trees, because duplicating existing data points makes them appear more important to the algorithm, leading to the model memorising specific instances rather than learning general patterns.
    - **Implementation:** The `imbalanced-learn` library provides `RandomOverSampler`.
3. **SMOTE (Synthetic Minority Over-sampling Technique):**
    - **Concept:** An oversampling technique that addresses the overfitting issue of random oversampling by **generating new, synthetic data points** for the minority class rather than just duplicating existing ones.
    - **Mechanism:** Uses **interpolation** to create new data points between existing minority class instances.
    - **Algorithm Steps:**
        1. Selects a minority class data point randomly.
        2. Finds its `k` nearest neighbors within the minority class (e.g., using a k-NN algorithm).
        3. Randomly chooses one of these `k` neighbors.
        4. Generates a new synthetic data point along the line segment connecting the original data point and its chosen neighbor, using a factor (randomly chosen between 0 and 1). This process continues until balance is achieved.
    - **Advantages:**
        - **Reduces bias**.
        - **Avoids simple duplication**, which helps to mitigate the overfitting problem associated with random oversampling.
    - **Disadvantages:**
        - **Cannot handle categorical data** directly, as interpolation doesn't make sense for discrete values.
        - Can be **slow for high-dimensional data** due to the underlying k-NN computations.
        - **Performance is sensitive to the `k` value** (number of neighbors). A low `k` might generate points only along specific lines, reducing variety, while a high `k` might generate points everywhere, diluting the original distribution.
        - **Sensitive to outliers**; an outlier in the minority class can lead to the generation of more noisy, synthetic outliers.
        - No guarantee that the **synthetically generated data points accurately represent the true underlying data distribution**.
    - **Implementation:** The `imbalanced-learn` library provides the `SMOTE` class. Many variants of SMOTE exist.
4. **Ensemble Methods (e.g., Balanced Random Forest):**
    - **Concept:** Modifies existing ensemble learning algorithms (like Random Forest) to handle imbalanced data more effectively.
    - **Mechanism (Balanced Random Forest):**
        - In a standard Random Forest, multiple decision trees are trained on bootstrapped samples of the data.
        - In a Balanced Random Forest, when creating each new sample for a decision tree, it ensures that the sample is **balanced** (e.g., by undersampling the majority class to match the minority class within that specific sample).
        - Each tree is trained on a balanced subset, and predictions are aggregated as usual.
    - **Benefits:** Helps the ensemble to give equal consideration to both classes.
    - **Implementation:** The `imbalanced-learn` library offers a `BalancedRandomForestClassifier`.
5. **Cost-Sensitive Learning:**
    - **Concept:** Modifies the **learning process** of the machine learning algorithm itself to account for class imbalance. This is achieved by assigning different "costs" to different types of errors.
    - **Two main approaches:**
        - **Class Weights:**
            - **Mechanism:** Assigns a higher weight to the minority class and a lower weight to the majority class during training. This tells the algorithm to penalise errors on the minority class more severely during its optimisation process (e.g., gradient descent).
            - **Benefits:** Directly influences the model to pay more attention to the minority class.
            - **Implementation:** Many scikit-learn classification algorithms (e.g., Logistic Regression, SVM, Decision Tree) have a `class_weight` parameter that allows you to specify these weights.
        - **Custom Loss Functions:**
            - **Mechanism:** Defines a tailored loss function where different types of errors (e.g., False Negatives vs. False Positives) have different penalties, reflecting their criticality in a specific problem (e.g., for a spam filter, False Positives might be more critical).
            - **Benefits:** Allows fine-grained control over how the model learns from errors, aligning with the problem's specific cost implications.
            - **Implementation:** More complex to implement than class weights, typically available in boosting algorithms like Gradient Boosting, XGBoost, and LightGBM, which allow passing custom objective functions.

## Additional Resources
The `imbalanced-learn` library's API reference provides a comprehensive list of other techniques and variants for handling imbalanced data, including combinations of oversampling and undersampling methods.