---
sidebar_position: 1
title: K‑Means Clustering — Concepts, Elbow Method, and Implementation
---

K-Means Clustering is a highly important and powerful **unsupervised machine learning algorithm** used to group similar items together. Unlike supervised learning, it does not require pre-labelled data to perform its task. It's widely used in various fields, such as clustering customers in business, students in education, or even images.

### Core Concepts and Working Principle

The K-Means algorithm operates in a series of steps to achieve its clustering objective:

1. **Deciding the Number of Clusters (K)**
    - A crucial initial step is to **determine the number of clusters (K)** that the algorithm should form. The K-Means algorithm itself cannot automatically infer this number from the data; it must be specified by the user.
    - This is often challenging, especially with high-dimensional data where visual inspection is not possible. The **Elbow Method** is typically used to address this (detailed below).
2. **Random Centroid Initialisation**
    - Once K is decided, the algorithm **randomly selects K data points** from the dataset to serve as the initial **centroids**. These centroids represent the initial "centre" of each cluster.
    - For instance, if K is 3, three random points are chosen as initial centroids.
3. **Assigning Points to Clusters**
    - This is a core step where each data point is assigned to the cluster whose **centroid is closest to it**.
    - The **Euclidean distance** is typically used to measure the proximity between a data point and a centroid.
    - For each data point, its distance to every centroid is calculated, and the point is then assigned to the cluster associated with the centroid that has the **minimum distance**.
    - **Euclidean Distance Formula:** For two points (x1, y1) and (x2, y2), the distance is `sqrt((x2 - x1)^2 + (y2 - y1)^2)`. Importantly, this formula can be generalised for **any number of dimensions** using the dot product `np.dot(b-a, b-a)` for vectors `a` and `b`. This generalisation is why K-Means is powerful and works beyond 2D or 3D datasets.
4. **Moving Centroids**
    - After all points have been assigned to clusters, the **centroids are recalculated**. Each centroid is moved to the **mean position of all the data points currently assigned to its cluster**.
    - For example, if all points in a blue cluster have a certain CGPA and IQ, the new blue centroid will be the average CGPA and average IQ of those points.
5. **Checking for Convergence**
    - Steps 3 and 4 (assigning points and moving centroids) are repeated iteratively.
    - The algorithm **converges** (stops) when the positions of the centroids **no longer change significantly** between iterations, or when a maximum number of iterations is reached. If the centroids have moved, the process continues; if they remain the same, the algorithm has finished.

### Determining the Optimal Number of Clusters (The Elbow Method)

Since K-Means requires specifying the number of clusters (K) beforehand, the **Elbow Method** is a popular technique to help determine an appropriate K value.

1. **The Concept:** The Elbow Method plots the number of clusters (K) against a metric called **Within-Cluster Sum of Squares (WCSS)**, also known as **Inertia**.
2. **WCSS (Within-Cluster Sum of Squares):**
    - WCSS measures the sum of the squared distances between each data point and its assigned cluster's centroid.
    - For a single cluster, WCSS is the sum of squared distances of all points to that cluster's centroid.
    - For multiple clusters, the total WCSS is the sum of WCSS values for each individual cluster.
3. **The Elbow Curve:**
    - You typically run the K-Means algorithm for a range of K values (e.g., from 1 to 10 or 20).
    - For each K, you calculate the WCSS and plot it.
    - The plot (called the "Elbow Curve") typically shows that as the number of clusters (K) increases, the WCSS value **decreases significantly at first**. This is because more clusters mean points are closer to their respective centroids.
    - However, beyond a certain point, the **rate of decrease in WCSS slows down considerably**, forming an "elbow" shape on the graph.
4. **Identifying the Elbow Point:** The "elbow point" is the optimal K value. It represents the point of diminishing returns, where adding more clusters does not significantly reduce the WCSS, meaning the clusters are already well-defined. It's like finding a point on a mountain where the descent rate slows down, making it safer to stop.

### Practical Implementation (Python)

The sources detail how to implement K-Means from scratch in Python:

- **Class Structure:** Create a `KMeans` class with a constructor that takes `n_clusters` (defaulting to 2) and `max_iter` (maximum iterations, defaulting to 100). It also initialises `self.centroids` to `None`.
- **`fit` Method:** This is the main method that takes the data (`X`) as input.
    - **Random Centroid Selection:** It randomly samples `n_clusters` unique indices from the data and uses the corresponding data points as initial centroids.
    - **Iterative Process:** A loop runs up to `max_iter` times.
        - **`assign_clusters` Function:** This function calculates the Euclidean distance between each data point and every centroid. It then assigns each point to the cluster of the nearest centroid, returning a `cluster_group` (e.g., a NumPy array indicating which cluster each row belongs to).
        - **`move_centroids` Function:** This function updates the centroid positions. It identifies all points belonging to a specific cluster, calculates the mean of those points along each feature (column-wise), and sets this mean as the new centroid for that cluster.
        - **Convergence Check:** Before each iteration (or at the end of each), it compares the `new_centroids` with the `old_centroids`. If they are identical (e.g., using `np.array_equal`), the loop breaks as the algorithm has converged.
- **Final Output:** The algorithm returns the `cluster_group` after convergence, indicating the cluster assignment for each data point.

### Applications and Examples

The K-Means algorithm is demonstrated with practical examples:

- **Student Clustering:** A dataset of 200 students with their CGPA (Cumulative Grade Point Average) and IQ values is used. K-Means is applied to group students into distinct clusters (e.g., based on intelligence and hard work), allowing for tailored placement strategies or academic support.
    - **Example Clusters:** Students with high CGPA and high IQ (intelligent, hardworking), high CGPA but lower IQ (very hardworking), high IQ but lower CGPA (smart but lazy), and low CGPA and low IQ (need more effort).
- **Higher Dimensions:** The sources demonstrate that K-Means works perfectly in **3D** and higher dimensions, making it highly versatile for real-world datasets that often have many features. The same code logic applies regardless of the number of dimensions.

As a suggested task, users are encouraged to implement the Elbow Method themselves by calculating the WCSS (Inertia) at each step.