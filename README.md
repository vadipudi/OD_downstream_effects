# Impact of Outlier Detection on Machine Learning Models

## Overview

**Abstract:**  
When fitting machine learning models on datasets, there is a possibility of mistakes occurring with overfitting due to outliers. Such mistakes can lead to incorrect predictions and diminished model usefulness. Outlier detection is conducted as a precursor step to avoid these errors and improve model performance. This study compares how different outlier detection methods impact regression, classification, and clustering tasks. Multiple outlier detection algorithms were applied to clean various datasets; the cleaned data were then used to train downstream models. Performance with and without outlier removal was compared to identify trends. The study found that while supervised tasks (regression and classification) benefit only marginally, unsupervised clustering tasks can see considerable improvement—most notably when using Isolation Forest (IForest) and Principal Component Analysis (PCA).

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
  - [Supervised Tasks](#supervised-tasks)
  - [Clustering Tasks](#clustering-tasks)
- [Datasets](#datasets)
- [Outlier Detection Methods](#outlier-detection-methods)
- [Downstream Tasks](#downstream-tasks)
- [Performance Metrics](#performance-metrics)
- [Repository Structure](#repository-structure)
- [Usage Instructions](#usage-instructions)
- [Installation](#installation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Methodology

### Supervised Tasks

For regression and classification:

1. **Data Splitting:**  
   Each dataset is split into training and testing sets at varied sizes (e.g., 400, 600, 800, 1000, and 1200 samples for training).
2. **Preprocessing:**  
   - Data is normalized (min-max scaling to a [0, 1] range) to handle differences in feature scale.
   - Categorical features are encoded: nominal values via one-hot encoding and ordinal/time-based features via appropriate transformations (cyclical encoding for time features).
3. **Outlier Detection:**  
   Class-specific outlier detection is performed on training data by applying various OD algorithms. For each class (or overall, where applicable), outlier scores are computed and a contamination threshold (ranging from 0.05 to 0.35) is applied to remove outliers.
4. **Model Training:**  
   Cleaned training data is used to train regressors (e.g., Linear Regression, Ridge, Decision Trees, Random Forests, SVR) or classifiers (e.g., Logistic Regression, Decision Trees, Random Forest, SVM, KNN).
5. **Evaluation:**  
   The trained model is evaluated on the test set and performance metrics (e.g., rMSE, R², MAE, Accuracy, Precision, Recall, F1-score) are recorded.

### Clustering Tasks

For clustering, the process is similar except that the full dataset is cleaned via outlier detection and then fed into clustering algorithms. Ground truth labels (when available) allow comparison using metrics such as the silhouette score and adjusted Rand index (ARI).

## Datasets

This study uses 13 unique datasets sourced from the UCI Machine Learning Repository and OpenML. They include datasets for regression, classification, and clustering (with classification and clustering datasets overlapping to enable ground truth comparisons). In total, additional noisy versions of these datasets were created by adding impulse noise to simulate sensor errors.

_A sample summary table might look like:_

| Dataset                | Task                   | Features | Instances | Target        | Notes                     |
|------------------------|------------------------|----------|-----------|---------------|---------------------------|
| Abalone                | Regression             | 8        | 4,177     | Age           | Encoded categorical data  |
| Bike Share Demand      | Regression             | 11       | 17,379    | Count         | Contains time features    |
| Covertype Partial      | Classification/Clustering | 54   | 8,000     | Cover Type    | Multi-class, high-dim     |
| ...                    | ...                    | ...      | ...       | ...           | ...                       |

For further details, please refer to the publication or thesis document.

## Outlier Detection Methods

The following outlier detection algorithms were used in this study, along with key parameters:

- **K Nearest Neighbors (KNN) and Local Outlier Factor (LOF):**  
  Parameter A: Number of neighbors.

- **Isolation Forest (IForest):**  
  Parameter A: Number of estimators;  
  Parameter B: Max features (range defined as min, small, medium, large, max).

- **Principal Component Analysis (PCA):**  
  Combined with kNN for outlier detection;  
  Parameter A: Number of neighbors (via kNN);  
  Parameter B: Number of components (up to one less than the feature count).

- **Kernel Density Estimation (KDE) and Mahalanobis Distance:**  
  Other probabilistic/distance-based approaches implemented using SciKit-Learn or custom code.

A contamination threshold (0.05–0.35) is used to determine the proportion of data considered outliers.

## Downstream Tasks

### Regression Models
- Multiple Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Elastic Net Regression  
- Random Forest Regression  
- Support Vector Regression

### Classification Models
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Classifier  
- K Nearest Neighbors (KNN)  
- Gradient Boosting

### Clustering Algorithms
- K-Means  
- Gaussian Mixture  
- Agglomerative Clustering  
- DBSCAN  
- HDBSCAN

## Performance Metrics

For each downstream task, performance is measured by:
- **Regression:**  
  *Root Mean Squared Error (rMSE)*, *Mean Absolute Error (MAE)*, *Adjusted R-squared*

- **Classification:**  
  *Accuracy*, *Precision*, *Recall*, *F1-score*

- **Clustering:**  
  *Silhouette Score*, *Adjusted Rand Index (ARI)*

Relative changes with and without outlier removal are quantified to assess the impact of each OD method.


## Usage Instructions

1. **Download Data:**  
   Place your datasets in the proper subdirectories under `data/` or follow the instructions in the README.

2. **Preprocessing:**  
   Use the provided notebooks and modules in `src/preprocessing/` (if present) to encode, normalize, and add simulated noise to datasets.  
   _Tip: Review `notebooks/0_data_exploration.ipynb` for an overview._

3. **Run Experiments:**  
   - For Regression: `python src/pipeline/run_regression.py`
   - For Classification: `python src/pipeline/run_classification.py`
   - For Clustering: `python src/pipeline/run_clustering.py`

4. **Analyze Metrics:**  
   Results and metrics are saved in the `results/` folder.  
   Use notebooks in `notebooks/` (e.g., `3_clustering_analysis.ipynb`) to visualize performance trends.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/thesis-outlier-detection.git
   cd thesis-outlier-detection
   ```
# OD_downstream_effects
