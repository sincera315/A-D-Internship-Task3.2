# A-D-Internship-Task3.2
# Feature Selection Techniques with Synthetic Dataset

This Jupyter Notebook demonstrates various feature selection techniques on a synthetic dataset. The notebook includes data generation, preprocessing, feature selection using different methods, model training, and evaluation. Additionally, it compares the selected features visually.

## Table of Contents

- [Introduction](#introduction)
- [Setup and Requirements](#setup-and-requirements)
- [Data Generation](#data-generation)
- [Feature Engineering](#feature-engineering)
- [Data Splitting](#data-splitting)
- [Feature Selection Methods](#feature-selection-methods)
  - [Univariate Feature Selection](#univariate-feature-selection)
  - [Recursive Feature Elimination (RFE)](#recursive-feature-elimination-rfe)
  - [Random Forest Feature Importance](#random-forest-feature-importance)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Feature Selection Comparison](#feature-selection-comparison)
- [Conclusion](#conclusion)

## Introduction

This project demonstrates the application of various feature selection techniques on a synthetic dataset generated using scikit-learn. The selected features are then used to train a machine learning model, and the performance is evaluated.

## Setup and Requirements

To run this notebook, you need to have the following libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

## Data Generation
The synthetic dataset is generated using the `make_classification` function from scikit-learn. It includes 1000 samples and 20 features, of which 10 are informative and 5 are redundant.

## Feature Engineering
Polynomial and interaction features are generated using `PolynomialFeatures` from scikit-learn.

## Data Splitting
The dataset is split into training and testing sets using an 80-20 split.

## Feature Selection Methods

### Univariate Feature Selection
Univariate feature selection is performed using `SelectKBest` with the `chi2` score function. The top 10 features are selected based on their chi-squared statistics.

### Recursive Feature Elimination (RFE)
RFE is used with a `LogisticRegression` estimator to recursively eliminate features until the top 10 features are selected.

### Random Forest Feature Importance
A `RandomForestClassifier` is trained, and the feature importances are used to select the top 10 features.

## Model Training and Evaluation
A `RandomForestClassifier` is trained on the selected features, and the model's performance is evaluated using accuracy score and classification report.

## Feature Selection Comparison
A line plot is created to compare the features selected by each method. This visualization helps in understanding the differences and overlaps between the feature selection techniques.

## Conclusion
This notebook demonstrates the importance of feature selection in machine learning. By comparing different feature selection methods, we can better understand which features are most important for our models and improve their performance.
"""
