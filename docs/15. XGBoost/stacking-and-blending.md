---
sidebar_position: 4
title: Stacking and Blending
---

## Introduction to Stacking

- Stacking is an **ensemble machine learning technique**, similar to boosting and bagging.
- It is particularly famous and frequently used in winning solutions for Kaggle competitions.
- The video aims to explain the intuition behind stacking, the algorithm, and its implementation in scikit-learn.

## Stacking vs. Voting Ensembles

- Stacking is an **extension of the voting ensemble idea**.
- In voting ensembles (for regression problems), multiple regression algorithms (e.g., Linear Regression, Random Forest Regressor, K-Nearest Neighbours Regressor) are trained on the same dataset.
- During prediction, a new data point is fed to all trained models, and their individual predictions are averaged to produce the final output.
- For classification problems, voting ensembles use a **majority count** of predictions.
- Stacking takes this a step further: instead of simply averaging or taking a majority vote, the **outputs of base learners are used as inputs for another machine learning algorithm**, called a **meta-model**, which is then trained.

## The Stacking Process (Basic Idea)

1. **Step 1: Train Base Models**
    - Take your dataset (e.g., with columns like CGPA, IQ, Package).
    - Train multiple base algorithms (e.g., Linear Regression, Decision Tree Regression, K-Nearest Neighbours Regression) on this dataset.
    - After this step, you have trained base models.
2. **Step 2: Generate New Data for Meta-Model**
    - Run these trained base models on the *same* training dataset.
    - For each student/row in the dataset, each base model will produce a prediction.
    - These predictions (e.g., Linear Regression prediction, Decision Tree prediction, KNN prediction) become **new input columns** for a new dataset.
    - The original target variable (e.g., 'Package') remains the target variable for this new dataset.
    - This new dataset is formed by the predictions of the base models as features, and the original target variable.
3. **Step 3: Train the Meta-Model**
    - Train a **meta-model** (e.g., Random Forest Regressor) on this newly generated dataset.
    - This completes the training phase.
4. **Prediction Phase**
    - When a new data point comes for prediction (e.g., a student with CGPA 7.7 and IQ 100):
        - It is first sent to all the trained base models, which produce their individual predictions.
        - These predictions are then fed as input to the trained meta-model.
        - The meta-model makes the final prediction.

## Key Differences from Bagging and Boosting

- **Base Model Diversity**: In stacking, the **base models can be of different types** (e.g., Linear Regression, Decision Tree, KNN). In contrast, bagging and boosting typically require base models of the same type.
- **Output Utilisation**: In stacking, the base models' outputs are **used to train a new model (meta-model)**. In bagging and boosting, base model outputs are directly used for majority voting, mean calculation, or weighted sums.

## The Overfitting Problem in Basic Stacking

- A significant problem in the basic stacking approach is **overfitting**.
- This occurs because the base models are trained on a dataset, and then predictions are made on the *same* dataset to generate inputs for the meta-model.
- If the base models are prone to overfitting (e.g., Decision Trees), their overfitted outputs will lead to the meta-model also becoming overfitted, potentially causing the entire model to fail.

## Solutions to Overfitting: Blending and K-Fold Stacking
There are two main methods to address the overfitting problem in stacking:

1. **Hold-out Method (Blending)**
    - This method is referred to as **blending**.
    - **Process**:
    - The original dataset is first divided into two parts: `D_train` (e.g., 80% of data) and `D_test` (e.g., 20% of data).
    - `D_train` is then further divided into two sub-parts: `D_train1` (e.g., 60% of `D_train`) and `D_validation` (e.g., 20% of `D_train`).
    - Step 1: Train Base Models — The base models (e.g., Linear Regression, Decision Tree, KNN) are trained only on `D_train1`.
    - Step 2: Generate New Data for Meta-Model — The trained base models are then used to make predictions on the `D_validation` set.
            - These predictions (e.g., `LR_pred`, `DT_pred`, `KNN_pred`) form the new input columns for a new dataset.
            - The actual target values from `D_validation` are used as the target for this new dataset.
            - This new dataset will have the same number of rows as `D_validation`.
        - **Step 3: Train Meta-Model**: The meta-model (e.g., Random Forest Regressor) is trained on this newly created dataset.
        - **Prediction Phase**: During testing, new data points are first passed through the trained base models, then their outputs are fed to the meta-model for final prediction.
    - **Benefit**: This design eliminates the overfitting problem because base models are trained on one subset (**D_train1**) and their predictions for the meta-model are generated on a separate, unseen subset (**D_validation**).
2. **K-Fold Method (Stacking)**
    - This method is referred to as **stacking** (though some people use "stacking" broadly for both).
    - **Process**:
    - The original dataset is divided into `D_train` (e.g., 800 rows) and `D_test` (e.g., 200 rows).
    - `D_train` is then divided into K equal "folds" (e.g., K = 4, so 4 folds of 200 rows each).
        - **Step 1: Train Base Models and Generate New Data** (this is the tricky part):
            - For each base model (e.g., Linear Regression):
                - It is trained **K times**.
                - In each iteration, **K-1 folds** are used for training, and the remaining **1 fold** is used for prediction.
                - The predictions from that held-out fold are stored.
                - Example (with K=4 and Linear Regression):
                    - Train LR on Folds 1, 2, 3; predict on Fold 4.
                    - Train LR on Folds 1, 2, 4; predict on Fold 3.
                    - Train LR on Folds 1, 3, 4; predict on Fold 2.
                    - Train LR on Folds 2, 3, 4; predict on Fold 1.
                - By concatenating these predictions, you get a full set of predictions (e.g., 800 predictions for `LR_pred`) for the entire `D_train` dataset.
            - This process is repeated for all base models (e.g., Decision Tree, KNN), resulting in multiple columns of predictions (e.g., `LR_pred`, `DT_pred`, `KNN_pred`), each containing 800 predictions.
            - These prediction columns, along with the original `D_train` target variable, form the new dataset for the meta-model (e.g., 800 rows, 4 columns).
            - **Note**: In this example, **9 base models** were trained in total (3 base models * 3 times each, as one model was not mentioned but it should be 4 times for each model if K=4 as per explanation).
        - **Step 2: Train the Meta-Model**: The meta-model (e.g., Random Forest Regressor) is trained on this newly created dataset.
    - **Step 3: Final Base Models for Prediction**: Crucially, for the final prediction phase, you do not use the K individual base models trained in Step 1. Instead, you train new instances of each base model on the entire `D_train` dataset. This provides one final, robust version of each base model.
        - **Prediction Phase**: During testing, new data points are passed through these newly trained final base models, their outputs are fed to the meta-model, and the meta-model gives the final output.
    - **Advantage**: This method ensures that the meta-model is trained on predictions that were made on data *unseen* by the base models during their specific training folds, thus mitigating overfitting.

## Complex Architectures in Stacking

- Stacking can involve more complex, multi-layered architectures beyond simple one-layer base models + one meta-model.
- For instance, one could have a first layer of base models, whose outputs feed into a second layer of models (which are themselves machine learning models), and then the outputs of the second layer feed into the final meta-model.
- In such multi-layered stacking, if using the blending approach, you would need to **divide the training data into more parts** (e.g., three parts for a two-layer architecture) to ensure that each layer's models are trained and make predictions on unseen data for the next layer.
- These complex architectures are common in winning Kaggle solutions, often involving a large number of models across multiple layers.

## Scikit-learn Implementation (StackingClassifier)

- Scikit-learn offers `StackingClassifier` (and `StackingRegressor`) for stacking.
- It requires **scikit-learn version 0.22** or higher.
- **Key Parameters**:
    - `estimators`: A list of tuples, where each tuple contains a name and an instance of a base model. Hyperparameter tuning for base models can be done here.
    - `final_estimator`: The meta-model. Hyperparameter tuning for the meta-model can also be done here.
    - `cv`: Specifies the number of folds for cross-validation (K-Fold method).
    - `stack_method`: Controls how prediction probabilities or decision functions are used as inputs for the meta-model in classification problems.
    - `passthrough`: A boolean parameter. If `True`, the **original input features (X_train)** are also passed as input to the meta-model, in addition to the base models' predictions. Generally, `passthrough=False` is used, meaning only the base model predictions are fed to the meta-model.
- **Example**: A `StackingClassifier` can be built with `RandomForestClassifier`, `KNeighborsClassifier`, and `GradientBoostingClassifier` as base models, and `LogisticRegression` as the meta-model.