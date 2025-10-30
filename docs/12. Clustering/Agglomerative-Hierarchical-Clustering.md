---
sidebar_position: 2
title: Agglomerative Hierarchical Clustering 
---

### Agglomerative Hierarchical Clustering: Overview

- Hierarchical clustering is a clustering algorithm discussed as an alternative to K-means clustering.
- Other clustering methods exist because K-means has several problems.
- **K-means relies heavily on distances to centroids**.
- **Problems with K-means**:
    - **Fails on non-spherical datasets**: It struggles with data that doesn't have well-defined, spherical boundaries, such as concentric circles, "moons" shaped datasets, or unevenly sized clusters.
    - **Assigns every point to a cluster**: K-means will assign every data point to a cluster, even noise or outliers, which may not be desirable.
    - **Best performance on spherical/circular datasets**: K-means performs best on data where clusters are spherical or circular and well-separated in higher dimensions.
- Other clustering methods, like Hierarchical Clustering and Density-Based Spatial Clustering of Applications with Noise (DBSCAN), address these limitations. DBSCAN, for example, clusters based on density and can identify outliers.

### Types of Hierarchical Clustering

There are two main types of Hierarchical Clustering:

1. **Agglomerative Hierarchical Clustering**: This is the more commonly used type and the main focus of the video.
2. **Divisive Hierarchical Clustering**: This type is not as widely used.

### Agglomerative Hierarchical Clustering (AGNES)

- **Intuition**:
    - Starts by assuming **each data point is its own cluster**.
    - Iteratively **merges the two closest clusters** (or points) into a new, single cluster.
    - This merging process is recorded in a **tree-like structure called a Dendrogram**.
    - The process continues until only one large cluster remains, encompassing all data points.
- **Dendrogram**:
    - A dendrogram is a **tree-like structure** that records the order and distance of merges.
    - It shows the **hierarchy of cluster formation**.
    - **Determining the Number of Clusters**: The dendrogram is used to decide how many clusters to form.
        - To find the optimal number of clusters, one looks for the **longest vertical line** in the dendrogram that is not intersected by any horizontal line.
        - Cutting the dendrogram at this point (horizontally) will reveal the desired number of clusters (indicated by the number of vertical lines intersected).
        - The vertical distances in the dendrogram represent the inter-cluster similarity or distance between merged clusters.
- **Agglomerative Algorithm Steps**:
    1. **Initialise the Proximity Matrix**: If there are `N` points, an `N x N` matrix is created, storing the distance between every pair of points. The diagonal elements (distance of a point to itself) are zero.
    2. **Each point is a cluster**: Initially, every data point is considered an individual cluster.
    3. **Iterative Merging Loop**:
        - **Find Closest Clusters**: Identify the two closest clusters (or points) based on their distance in the proximity matrix.
        - **Merge Clusters**: Combine these two closest clusters into a new single cluster.
        - **Update Proximity Matrix**: Re-calculate and update the distances in the proximity matrix for the newly formed cluster with respect to all other existing clusters/points. This is the most complex part of the algorithm, as the method for calculating inter-cluster distance defines different types of agglomerative clustering.
        - **Repeat** until only one cluster remains.

### Divisive Hierarchical Clustering (DIANA)

- **Intuition**:
    - Starts with the **assumption that all points belong to a single cluster**.
    - It then iteratively **breaks down the large cluster** into smaller clusters.
    - The process continues until each point becomes an individual cluster.
- **Comparison with Agglomerative**: Divisive clustering works in the exact opposite way to agglomerative clustering.
- **Why Agglomerative is Preferred**: Merging clusters based on distance is generally considered easier than breaking them apart.

### Types of Agglomerative Hierarchical Clustering (Linkage Methods)

The way the distance between two clusters (inter-cluster similarity) is calculated determines the type of agglomerative clustering.

1. **Single Link (Minimum Linkage)**:
    - **Method**: The distance between two clusters is defined by the **shortest distance between any two points**, one from each cluster.
    - **Benefit**: Works well when there are clear gaps between clusters.
    - **Disadvantage**: **Highly susceptible to outliers or noise** in the data.
2. **Complete Link (Maximum Linkage)**:
    - **Method**: The distance between two clusters is defined by the **largest (maximum) distance between any two points**, one from each cluster.
    - **Benefit**: **Effectively handles outliers or noise** in the data.
    - **Disadvantage**: May break down large clusters into smaller ones if a cluster is significantly larger than others.
3. **Average Link (Group Average)**:
    - **Method**: The distance between two clusters is the **average of all pairwise distances** between points from both clusters.
    - **Purpose**: Aims to strike a balance between the single-link and complete-link methods, mitigating their individual disadvantages.
4. **Ward's Method**:
    - **Method**: Focuses on **minimising the variance within clusters**. It calculates the distance between two clusters based on how much the sum of squares of deviations from the centroid increases when they are merged.
    - **Purpose**: Also aims to provide a balance between single and complete linkage.
    - **Default**: This is often the **default linkage method** in implementations like `scikit-learn`'s `AgglomerativeClustering`.

### Parameters for Agglomerative Clustering (Python Implementation)

The `AgglomerativeClustering` class in Python's `scikit-learn` library has four key parameters:

- **`n_clusters`**: Specifies the desired number of clusters to form.
- **`affinity`**: Determines the distance metric used for inter-cluster distance calculation (e.g., Euclidean, Manhattan, Cosine). Euclidean is the default.
- **`linkage`**: Specifies the linkage criteria (type of agglomerative clustering) to use (e.g., 'ward', 'complete', 'average', 'single'). 'Ward' is the default.
- **`distance_threshold`**: A threshold for the distance between clusters; if the distance is below this, merging occurs. If specified, `n_clusters` cannot be set.

### Practical Application

- The process involves loading data (e.g., customer data with annual income and spending score).
- Then, using libraries like SciPy, a **dendrogram is generated** from the data.
- The dendrogram is then analysed (as described above) to **determine the optimal number of clusters**.
- Finally, the `AgglomerativeClustering` class is used with the chosen parameters (`n_clusters`, `affinity`, `linkage`) to perform the clustering and obtain cluster labels for each data point. The results can then be visualised.

### Benefits of Hierarchical Clustering

- **Wide Applicability**: Can cluster a wide variety of datasets, overcoming the limitations of K-means on certain data types. The different linkage methods allow for flexibility with various data structures.
- **Dendrogram Provides Hierarchy Information**: The dendrogram provides a **complete hierarchy** of how clusters are formed, showing which points are closest at each stage. This detailed information about proximity is not available in K-means.

### Limitations of Hierarchical Clustering

- **Scalability Issues (Not suitable for large datasets)**: The biggest limitation is that it **cannot be used on large datasets**.
    - This is because the algorithm requires creating and storing an `N x N` proximity matrix, where `N` is the number of data points.
    - For example, with 10 million (10^7) data points, the matrix would require 10^14 elements, which translates to terabytes of information, making it impractical to store in RAM.