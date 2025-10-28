---
sidebar_position: 4
---

# Handling Missing Values

Missing values are one of the most common data quality issues in machine learning datasets. Proper handling of missing values is crucial for building robust models and ensuring accurate predictions.

## Understanding Missing Data

### Types of Missing Data

#### 1. MCAR (Missing Completely At Random)
- **Definition**: Missing values occur randomly and are not related to any observed or unobserved variables
- **Example**: Data loss due to technical issues, random survey non-responses
- **Characteristics**: No pattern in missingness, completely random

#### 2. MAR (Missing At Random)
- **Definition**: Missing values depend on observed variables but not on the missing values themselves
- **Example**: Optional fields in forms, income missing for unemployed individuals
- **Characteristics**: Missingness can be predicted from other variables

#### 3. MNAR (Missing Not At Random)
- **Definition**: Missing values depend on the missing values themselves
- **Example**: High-income individuals refusing to report income, patients dropping out due to severe symptoms
- **Characteristics**: Most problematic type, creates bias in analysis

## Strategies for Handling Missing Values

### 1. Complete Case Analysis (Listwise Deletion)

**What it is**: Removing entire rows that contain any missing values

**When to use**:
- Data is MCAR (Missing Completely At Random)
- Less than 5% of data is missing
- Missing values are truly random

**Advantages**:
- Easy to implement (no data manipulation required)
- Preserves variable distribution (if data is MCAR)
- No assumptions about missing data patterns

**Disadvantages**:
- Can exclude large fraction of data from analysis
- Excluded observations may be informative
- Models in production won't know how to handle missing data
- Reduces statistical power

### 2. Handling Mixed Data

**Example**: Cabin numbers like "B5", "A23" contain both letters and numbers

**Solution**: Separate into different columns
- Extract numeric part: 5, 23
- Extract character part: B, A
- Handle missing values separately for each component

## Univariate Imputation Methods

These methods fill missing values using only information from the same column.

### 1. Mean/Median Imputation

**Mean Imputation**:
- Use when data is normally distributed
- Replace missing values with column mean
- Simple and fast

**Median Imputation**:
- Use when data is skewed
- Replace missing values with column median
- More robust to outliers

**Advantages**:
- Easy to implement
- Preserves sample size
- Good for small amounts of missing data

**Disadvantages**:
- Changes data distribution
- Reduces variance
- Alters correlations between variables
- May introduce bias

### 2. Arbitrary Imputation

**Method**: Replace missing values with predefined arbitrary values
- For numerical: Use 0, -1, or 999
- For categorical: Use "Unknown" or "Missing"

**When to use**: When data is not missing at random (MNAR)

**Disadvantages**:
- Changes probability distribution function (PDF)
- Alters variance and covariance
- May not reflect true data patterns

### 3. End of Distribution Imputation

**For Normal Data**:
- Replace with mean ± 3 × standard deviation
- Places values at extreme ends of distribution

**For Skewed Data**:
- Use Q1 - 1.5 × IQR or Q3 + 1.5 × IQR
- IQR = Interquartile Range

**When to use**: When data is not missing at random (MNAR)

**Disadvantages**:
- Changes PDF, variance, and covariance
- Creates artificial outliers
- May not represent true data patterns

### 4. Random Imputation

**Method**: Fill missing values with random values from the same column

**Advantages**:
- Preserves original distribution
- Good for linear models
- Maintains variance

**Disadvantages**:
- Memory intensive for deployment
- Requires storing original training set
- May not capture true relationships

### 5. Missing Indicator

**Method**: Create a separate binary column indicating missing values
- 1 if value is missing, 0 if present
- Keep original column with imputed values

**Advantages**:
- Preserves information about missingness
- Useful for tree-based algorithms
- Can improve model performance

## Multivariate Imputation Methods

These methods use relationships between multiple columns to fill missing values.

### 1. KNN Imputer

**How it works**:
1. Find k nearest neighbors for each missing value
2. Use their values to impute missing data
3. Calculate distance using available features only

**Distance Formula**:
```
nan_euclidean_distance = √((x2-x1)² + (y2-y1)²) × (total_columns / available_columns)
```

**Weighting Options**:
- **Uniform**: Average of k nearest neighbors
- **Distance**: Weighted average (closer neighbors have more influence)

**Advantages**:
- More accurate than univariate methods
- Uses relationships between variables
- Flexible (can choose k and weighting)

**Disadvantages**:
- Computationally expensive
- Requires storing entire dataset for deployment
- Sensitive to choice of k

### 2. Iterative Imputer (MICE)

**MICE**: Multivariate Imputation by Chained Equations

**How MICE works**:
1. Start with mean imputation for all missing values
2. For each column with missing values:
   - Remove all missing values from that column
   - Use other columns to predict missing values
   - Update the column with predicted values
3. Repeat process for all columns
4. Iterate until convergence or maximum iterations reached

**Advantages**:
- Uses all available information
- Handles different data types
- Can model complex relationships
- Produces multiple imputed datasets

**Disadvantages**:
- Computationally intensive
- Requires assumptions about data distribution
- May not converge for some datasets

## Best Practices for Missing Values

### 1. Data Exploration
- Visualize missing data patterns
- Identify which variables have missing values
- Understand the mechanism of missingness

### 2. Choose Appropriate Method
- **MCAR**: Complete case analysis or simple imputation
- **MAR**: Use multivariate imputation methods
- **MNAR**: Consider domain-specific approaches

### 3. Validation
- Compare imputed values with known patterns
- Check if imputation improves model performance
- Validate on holdout data

### 4. Documentation
- Document missing data patterns
- Record imputation methods used
- Keep track of assumptions made

### 5. Production Considerations
- Ensure imputation method works on new data
- Store imputation parameters for consistency
- Handle new missing patterns gracefully

## Common Pitfalls to Avoid

1. **Ignoring missing data**: Can lead to biased results
2. **Using wrong imputation method**: May introduce bias
3. **Not validating imputation**: May not improve model performance
4. **Forgetting production deployment**: Model must handle missing data in production
5. **Over-imputation**: Don't impute when data is truly missing for a reason

## Summary

Handling missing values is a critical step in the data preprocessing pipeline. The choice of method depends on:
- Type of missing data (MCAR, MAR, MNAR)
- Amount of missing data
- Data distribution and relationships
- Model requirements
- Production constraints

Always validate your approach and consider the implications for model performance and deployment.