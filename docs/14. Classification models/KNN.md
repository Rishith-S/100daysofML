---
sidebar_position: 2
title: KNN (K Nearest Neighbors)
---

### Introduction to K-Nearest Neighbors (KNN)

- KNN is a **simple, elegant, and very interesting machine learning algorithm**.
- Its logic is **very real-world based**, often explained by the quote: "You are the average of the five people you spend the most time with".
- It's a **lazy learner**, meaning it stores all training data during the training phase and performs computations only during the prediction phase.

### How KNN Works

1. **Define a 'K' value**: This determines how many nearest neighbours will be considered. For example, if K=3, the three nearest neighbours will be considered.
2. **Calculate Distances**: For a new query point, the algorithm calculates the distance to every other point in the training data.
    - **Euclidean distance** is typically used, but other distance metrics can also be employed.
3. **Sort and Select Nearest Neighbours**: All calculated distances are sorted, and the 'K' points closest to the query point are selected.
4. **Apply Majority Count (Democracy/Voting)**: The class label of the query point is determined by the majority class among its 'K' nearest neighbours. For instance, if two neighbours are class '1' and one is class '0', the query point is labelled as '1'.

### Application Example: Breast Cancer Dataset Classification

- KNN can be used for **classification problems**, such as predicting if breast cancer is present or not.
- **Data Preparation**:
    - Irrelevant columns (like ID or columns with missing values) are typically dropped.
    - The dataset is split into **training and testing sets** to train the model and then evaluate its performance on unseen data.
    - **Data Scaling is Crucial**:
        - Features in a dataset often exist on **different scales** (e.g., one column in tens, another in decimals).
        - If not scaled, features with larger values can **dominate the distance calculation**, making the distance metric unreliable.
        - A **Standard Scaler** is used to transform all column values to a similar scale, ensuring reliable distance calculations.
- **Model Training and Prediction**:
    - A KNN classifier object is created, specifying the number of neighbours (K).
    - The model is trained (`fit`) on the scaled training data (X_train, y_train).
    - Predictions (`predict`) are then made on the scaled test data (X_test).
- **Accuracy Score**: This metric is used to evaluate the performance of the classification model by comparing predicted values with actual values.

### Finding the Best 'K' Value

Choosing the optimal 'K' value is crucial for model performance and can be done using two approaches:

1. **Heuristic Approach**:
    - A common heuristic is to take the **square root of N** (number of observations/patients).
    - If N is 113, K could be 10 or 11. It's often advised to choose an **odd 'K'** value to avoid ties in majority voting.
2. **Experimental Approach (Cross-Validation)**:
    - This involves iterating through a **range of 'K' values** (e.g., from 1 to 15).
    - For each 'K', a KNN model is trained on the training data, and its accuracy is calculated on the test data.
    - The **accuracies are plotted against their corresponding 'K' values**.
    - The 'K' value that yields the **highest accuracy** is considered the best for that dataset. For example, a K value of 3 might yield the best accuracy.

### Decision Surfaces (Decision Boundaries)

- **Purpose**: A tool used to **visualise and understand how classification models work** by showing how the coordinate system is divided into different classification regions.
- **How it's Created**:
    1. **Plot training data** (e.g., GPA and IQ) with their respective classes (e.g., red and blue points).
    2. **Generate a grid of points** across the X and Y ranges of the training data.
    3. **Train the KNN model** using the training data.
    4. **Send each generated grid point to the trained KNN model** for prediction.
    5. **Colour the regions** based on the predicted class for each grid point (e.g., orange for class 0, blue for class 1).
- **Interpretation**:
    - Regions coloured by predicted classes are called **decision regions**.
    - The line or surface where the class prediction changes is called the **decision boundary**.
    - A new query point falling within an orange region would be predicted as class 0, and one in a blue region as class 1.

### Impact of 'K' on Model Behaviour: Overfitting vs. Underfitting

The choice of 'K' significantly influences whether a KNN model overfits or underfits the data:

- **Low 'K' Value (e.g., K=1 or 2)**:
    - Leads to **overfitting**.
    - The decision surface becomes **highly complex, jagged, or bumpy**, creating many small regions.
    - The model "memorises" or "over-learns" even minor variations and noise in the training data, including outliers.
    - This results in **high variance**, meaning the model's performance will vary significantly on different datasets.
- **High 'K' Value (e.g., K approaching the total number of observations, N)**:
    - Leads to **underfitting**.
    - The decision surface becomes **very smooth and simplistic**.
    - The model fails to capture the underlying patterns in the data and might simply predict the **majority class** regardless of the query point's features.
    - This results in **high bias** and poor performance on both training and test data.
- **Optimal 'K' Value**:
    - A value in between the extremes, found through experimentation (like cross-validation), is needed.
    - This optimal 'K' creates a **smoother, non-linear decision boundary** that truly represents the data without over- or under-fitting.

### Failure Cases/Drawbacks of KNN

KNN, while simple, has several limitations:

1. **Large Datasets**:
    - KNN is **very slow during the prediction phase**.
    - For every new query point, it calculates distances to all training points, sorts them, and applies majority count.
    - With 500,000 points, this can take several seconds per prediction, making it unsuitable for applications requiring low latency (e.g., online platforms).
2. **High-Dimensional Data (Curse of Dimensionality)**:
    - In very high-dimensional datasets, the **concept of distance becomes unreliable**.
    - Euclidean distance, which KNN heavily relies on, becomes less meaningful, leading to **unreliable results**.
3. **Outliers**:
    - KNN is **sensitive to outliers**.
    - Outliers can improperly influence the decision boundaries, leading to incorrect classifications and potential overfitting if the 'K' value is not appropriately tuned.
4. **Features on Different Scales**:
    - As discussed, if features are on different scales, the **distance metric will be unreliable**.
    - Features with larger numerical ranges will disproportionately influence the distance calculation.
    - **Data scaling is mandatory** to obtain reliable results.
5. **Imbalanced Datasets**:
    - If one class heavily outnumbers others (e.g., 98% "yes" points), KNN tends to be **biased towards the majority class**.
    - This can lead to poor predictive performance for the minority class.
6. **Lack of Interpretability (Black-Box Model)**:
    - KNN is good for **prediction** but **poor for inference**.
    - It cannot explain *why* a particular prediction was made or which features contributed most to the outcome (e.g., whether GPA or IQ had a greater impact on placement prediction). It's a **black-box model** in this regard.