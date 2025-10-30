---
sidebar_position: 1
title: Introduction to XGBoost
---

## 1. Introduction to XGBoost

- XGBoost (Extreme Gradient Boosting) is a highly **famous and widely used machine learning library**.
- It is prominently used in **Kaggle competitions** and **industry projects**.
- This video aims to provide a **detailed introduction and overview** of XGBoost, simplifying concepts from its original paper.
- The goal is to provide a comprehensive overview to help beginners understand its structure and features, overcoming potential overwhelm.

## 2. Machine Learning Algorithm Evolution

- **Definition of Machine Learning**: Learning from data by applying algorithms to datasets to identify patterns and make predictions.
- **Early Algorithms (1970s-1980s)**:
    - Examples: Naive Bayes, Linear Regression.
    - **Advantage**: Solved problems using machine learning.
    - **Major Disadvantage**: **Data-specific or scenario-specific**, performing well only on very specific types of data (e.g., Linear Regression for linear data, Naive Bayes for textual data). Not general-purpose.
- **Algorithms of the 1990s**:
    - Examples: **Random Forest, SVM, and Gradient Boosting**.
    - **Advantages**: More powerful performance-wise and more general, applicable to different data types.
    - **Disadvantages**:
        - Still struggled with **overfitting**.
        - Had **scalability problems**; became ineffective on larger datasets that emerged around the 2000s with the internet boom.
        - Issues with both **performance (metrics)** and **speed** on large datasets.

## 3. Emergence of XGBoost (2014)

- XGBoost was developed in **2014** to solve the **performance and speed problems** of previous algorithms, especially with large datasets.
- It quickly became the **"best machine learning algorithm out there"** for its performance on various and large datasets.

## 4. XGBoost: Not an Algorithm, but a Library

- Many mistakenly think XGBoost is an algorithm; however, it is a **library built on the existing Gradient Boosting algorithm**.
- The creator, **Tianqi Chen**, saw the potential in Gradient Boosting and aimed to improve it in terms of **performance and speed**.
- **XGBoost = Gradient Boosting + many software engineering optimisations**.
- It merges **machine learning concepts** (from Gradient Boosting) and **software engineering concepts** to create a powerful library.

## 5. Why Gradient Boosting as the Base?

- Tianqi Chen chose Gradient Boosting for XGBoost due to several factors:
    - **Flexibility**:
        - Supports **any differentiable loss function**, unlike other ML algorithms with specific loss functions.
        - Can handle **regression, classification, ranking, and even custom-defined problems**.
    - **Performance**: Generally provides **good results** on most datasets.
    - **Robustness**: Produces **robust results** with proper regularisation.
    - **Missing Value Handling**: Internally handles missing values very well.
    - Already being used by many to **win Kaggle competitions**.
- The main thought was to add **performance and scalability improvements** to make Gradient Boosting a "killer algorithm".

## 6. History of XGBoost Adoption

- **Early Days (2014)**:
    - Tianqi Chen published the paper **"XGBoost: A Scalable Tree Boosting System"**.
    - He himself participated in and **won the Higgs Boson Kaggle competition** using XGBoost. This brought initial notice to the algorithm.
- **Kaggle Boom (Around 2016)**:
    - In 2016, **16 out of 29 winning Kaggle submissions used XGBoost**, significantly boosting its popularity.
- **Open Source Days (Post-2016)**:
    - Tianqi Chen decided to **open-source XGBoost** to foster community contributions.
    - This led to rapid growth, adding **new features, optimizations, multi-platform support (different OS and programming languages)**, and extensive documentation.
    - It became a **default choice** for many Kaggle participants.
    - By 2023, XGBoost is considered a **must-know for data scientists and ML engineers** due to unmatched performance and speed.

## 7. Core Design Principles / Tianqi Chen's Concerns
When building XGBoost, Tianqi Chen focused on three major areas:

- **Performance**: Aimed for excellent performance, robustness, and avoidance of overfitting.
- **Speed**: Crucial for scalability and handling large datasets efficiently.
- **Flexibility**: Wanted the library to be usable by as many people as possible, regardless of programming language or operating system.

## 8. XGBoost Flexibility Features
XGBoost's flexibility is a key aspect of its success:

- **Cross-Platform Support**: Models can run on **any operating system** (Linux, Windows, Mac). Models trained on one OS can be run on another.
- **Multi-Language Support**:
    - Unlike most predictive models limited to Python, R, or Matlab, XGBoost provides **wrappers for many popular programming languages**.
    - Supported languages include **Java, Scala, Ruby, Python, R, Swift, Julia, C#, and C++**.
    - Allows **building a model in one language (e.g., Python) and loading/using it in another (e.g., Java)**, which is highly convenient for enterprise applications and avoids complex API setups.
- **Compatibility with Other Libraries**:
    - **Model Building**: Compatible with **NumPy, Pandas, Matplotlib, Scikit-learn**.
    - **Distributed Computing**: Compatible with **Spark, PySpark, Dask**.
    - **Model Interpretability**: Compatible with **Shap, Lime**.
    - **Model Deployment**: Compatible with **Docker, Kubernetes**.
    - **Workflow Management**: Compatible with **Apache Airflow, MLflow**.
- **Variety of Machine Learning Problems**:
    - Unlike algorithms limited to specific problem types (e.g., Linear Regression for regression, Logistic Regression for classification), XGBoost can solve **any type of ML problem**.
    - Supports **regression, binary classification, multi-class classification, time series forecasting, ranking problems (e.g., recommender systems), and anomaly detection**.
    - Allows users to define **custom loss functions and custom evaluation metrics** (due to its GBDT base, which supports any differentiable loss function).

## 9. XGBoost Speed Optimizations
XGBoost's speed is a result of numerous software optimizations:

- **Empirical Proof**: A comparison showed XGBoost training was **12-13 times faster** than Gradient Boosting on a synthetic dataset (5 seconds vs. 72 seconds for 10,000 rows, 200 columns). This means a 10-hour training job could take just 1 hour with XGBoost.
- **Parallel Processing**:
    - Although boosting is inherently sequential (models built one after another based on errors), **XGBoost applies parallel processing *within* the building of each individual decision tree**.
    - When finding the best split point for a node, **different features are processed in parallel**. For example, processing feature 1 to find its best split can happen simultaneously with processing feature 2.
    - This significantly speeds up tree construction, especially with many features.
    - Activated using the `n_jobs` hyperparameter (e.g., setting to -1).
- **Optimized Data Structures (Column Blocks)**:
    - Traditional ML algorithms store data row-wise.
    - XGBoost stores data in **column blocks**, where features are stored separately.
    - This structure enables efficient parallel processing by allowing different processor cores to operate on different column blocks simultaneously.
- **Cache Awareness**:
    - XGBoost efficiently uses the CPU's **cache memory** (a small, fast memory close to the CPU).
    - It stores frequently used information, such as **histogram bins** (used in Histogram-based training for numerical features), in the cache.
    - This reduces the need to constantly fetch data from slower RAM, speeding up training.
- **Out-of-Core Computing**:
    - Addresses the challenge of training models on datasets **larger than the available RAM**.
    - XGBoost can divide large datasets into **smaller chunks** and load them into RAM sequentially for processing.
    - This feature is activated by setting the `tree_method` hyperparameter to `'hist'`.
    - It works in tandem with **cache awareness** to store crucial information from each chunk in the cache for efficient training.
- **Distributed Computing**:
    - Allows training models across **multiple machines or nodes** (a cluster of computers).
    - The large dataset is divided into parts, and each node trains on its portion **in parallel**, with a master node aggregating results.
    - This is even faster than out-of-core computing for very large datasets, as multiple machines work concurrently.
    - Requires integration with external libraries like **Dask or Kubernetes**.
- **GPU Support**:
    - Leverages **Graphical Processing Units (GPUs)** for faster computation, especially for highly parallelisable tasks like histogram creation and split finding.
    - GPUs have many less powerful cores, suitable for small, parallel calculations.
    - Setting the `tree_method` hyperparameter to `'gpu_hist'` enables GPU acceleration, significantly speeding up the training process.
- The "Extreme" in XGBoost's name comes from these **"extreme" software optimizations** applied to Gradient Boosting.

## 10. XGBoost Performance Optimizations (Machine Learning Concepts)
Tianqi Chen also used advanced ML concepts to improve performance:

- **Regularised Learning Objective**:
    - Unlike traditional Gradient Boosting, XGBoost's **loss function inherently includes a regularisation term (L1 and L2)**.
    - This term, with parameters like `lambda` (for weights of leaves), helps **prevent overfitting** by simplifying the model during optimisation.
    - This leads to **more general models** that perform well on diverse datasets.
- **Handling Missing Values (Sparsity Aware Split Finding)**:
    - XGBoost can **natively handle missing values** without requiring pre-processing steps like imputation or removal.
    - It intelligently decides the best direction for a missing value (left or right branch of a split) by **calculating the gain for both possibilities** and choosing the one that maximises gain.
- **Efficient Split Finding**:
    - Addresses the slowness of **Exact Greedy Search** (which tests all possible split points for numerical features) on large datasets.
    - **Approximate Tree Learning**: Instead of testing every value, it **bins numerical features** (discretises continuous variables) to reduce the number of split candidates. This improves speed but might slightly reduce accuracy compared to exact search.
    - **Weighted Quantile Sketch**: To mitigate the accuracy loss from binning, XGBoost uses this technique to create bins. It **studies the data distribution** (e.g., using quantiles) and creates **bins that are denser where data points are concentrated and wider where data is sparse**. This makes the bins more representative and leads to more accurate trees.
- **Tree Pruning**:
    - XGBoost offers extensive options for **tree pruning**, a process that cuts or trims tree depth to **reduce complexity and prevent overfitting**.
    - Supports both **post-pruning** (growing the tree fully then trimming) and **pre-pruning** (deciding tree size during construction).
    - Includes a `gamma` hyperparameter to control whether a new branch should be formed, based on whether it leads to a significant loss reduction.
    - Effective pruning ensures better generalisation performance.

## 11. Other Gradient Boosting Libraries
While XGBoost is prominent, two other notable implementations are:

- **LightGBM (Microsoft Research)**:
    - A "lighter" alternative, often comparable or better than XGBoost in results.
    - Aimed at **faster training speeds, lower memory usage, and sometimes better accuracy**.
    - Also supports **parallel, distributed, and GPU learning**.
- **CatBoost (Yandex)**:
    - A high-performance open-source library.
    - Key feature: **Built-in support for categorical features**, which is a common challenge in ML.