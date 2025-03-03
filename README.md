# NYC Taxi Data Analysis

## Overview

This project analyzes New York City's yellow taxi trip data for January 2023, containing approximately 3 million rides. The dataset, sourced from TPEP/LPEP initiatives, offers insights into urban mobility, fare dynamics, and driver-passenger behavior. The analysis comprises feature engineering, regression, classification, clustering, and association rule mining.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Feature Engineering](#feature-engineering)
- [Regression Analysis](#regression-analysis)
- [Classification Analysis](#classification-analysis)
- [Clustering and Association Analysis](#clustering-and-association-analysis)
- [Results and Recommendations](#results-and-recommendations)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [References](#references)

## Introduction

This project applies Machine Learning techniques to analyze NYC taxi data. The main objectives are:

- Feature engineering and dimensionality reduction
- Regression analysis to predict the total fare amount
- Classification analysis to predict whether a tip was given
- Clustering to identify behavioral patterns
- Association rule mining to uncover relationships in categorical data

## Dataset Description

The dataset includes:
- **Temporal data:** Pickup and drop-off times
- **Spatial data:** Pickup and drop-off locations
- **Fare details:** Fare amount, tips, tolls, taxes, etc.
- **Trip information:** Distance, passenger count, payment method

## Feature Engineering

- **Data Cleaning:** Handled missing values using mode imputation and logical assumptions.
- **Outlier Detection and Removal:** Used the IQR method.
- **Dimensionality Reduction:** Applied PCA, SVD, Random Forest feature importance, and VIF analysis.
- **Feature Selection:** Extracted trip duration, pickup/drop-off days, and hours.

## Regression Analysis

- Applied **stepwise regression** to predict `total_amount`
- Evaluated using **R-squared (0.812)** and **MSE (0.188)**
- Key influencing factors: `trip_distance`, `payment_type`, `pickup_hour`

## Classification Analysis

- Predicted whether a **tip was given**
- Applied multiple classifiers:
  - Decision Tree (Pre/Post Pruned)
  - Logistic Regression
  - k-Nearest Neighbors (KNN)
  - Support Vector Machine (Linear, Polynomial, RBF)
  - Naïve Bayes
  - Random Forest (Bagging, Stacking, Boosting)
  - Neural Networks (MLP)
- **Best models:** Naïve Bayes (for tippers) and Random Forest Boosting (for non-tippers)
- Evaluated using accuracy, confusion matrices, ROC-AUC, and stratified K-fold cross-validation.

## Clustering and Association Analysis

- **K-means clustering** (Elbow & Silhouette methods) identified **9 clusters** related to:
  - Temporal patterns (rush hours)
  - Traffic conditions
  - Payment behavior (tipping habits)
- **Apriori association mining** found frequent categorical associations.

## Results and Recommendations

- Regression model accurately predicts fare amounts.
- Classification model effectively predicts tipping behavior.
- Clustering reveals distinct behavioral patterns.
- Association rule mining provides valuable business insights.

## Technologies Used

- **Programming Language:** Python
- **Libraries:** NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Machine Learning Techniques:** PCA, SVD, Random Forest, Regression, Classification, Clustering, Association Mining

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the analysis:
```bash
python analysis.py
```
