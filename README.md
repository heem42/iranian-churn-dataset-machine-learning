# Iranian Churn Dataset - Machine Learning Results


## Project Overview

This project analyzes the Iranian Churn Dataset to predict customer churn using 4 machine learning models.
- Logistic Regression
- K-Nearest Neighbors (with SMOTE oversampling)
- Decision Tree
- Random Forest. 

## Evaluation Methods

We evaluate our models using 3 methods
- Holdout Method
- Cross Validation
- Random Subsampling

## Steps to Run

```bash
pipenv install
pipenv run python src.py
```

## Output

```
ITEC3040 IRANIAN CHURN DATASET
Team: Data Miners
Members: Amen Abrham, Ibraheem Ahmed, Nuha Naushad, Samirah Mohammad
Data shape: (3150, 13)
Logistic Regression: {'C': 10}
K-Nearest Neighbors: {'n_neighbors': 9}
Decision Tree: {'max_depth': None}
Random Forest: {'max_depth': None, 'n_estimators': 50}


Evaluation Method 1) HOLDOUT METHOD (80/20 SPLIT)
                     Precision  Recall  F1-Score  ROC-AUC
Logistic Regression     0.8200  0.4141    0.5503   0.9240
K-Nearest Neighbors     0.7037  0.9596    0.8120   0.9809
Decision Tree           0.7732  0.7576    0.7653   0.8673
Random Forest           0.9213  0.8283    0.8723   0.9879


Evaluation Method 2) 10-FOLD CROSS-VALIDATION (Mean)
                    Precision  Recall F1-Score ROC-AUC
Logistic Regression    0.7380  0.4475   0.5522  0.9351
K-Nearest Neighbors    0.9185  0.9659   0.9415  0.9848
Decision Tree          0.7990  0.7915   0.7920  0.8908
Random Forest          0.9045  0.8143   0.8551  0.9815


Evaluation Method 3) RANDOM SUBSAMPLING
                    Precision  Recall F1-Score ROC-AUC
Logistic Regression    0.7778  0.4636   0.5791  0.9339
K-Nearest Neighbors    0.7167  0.9293   0.8087  0.9693
Decision Tree          0.7931  0.7778   0.7844  0.8869
Random Forest          0.8816  0.7949   0.8352  0.9810


Feature Importance on Churn using Random Forest
1. Complains: 18.8%
2. Seconds of Use: 12.6%
3. Status: 12.2%
4. Subscription  Length: 11.9%
5. Frequency of use: 11.0%
6. Customer Value: 7.7%
7. Distinct Called Numbers: 7.7%
8. Call  Failure: 6.0%
9. Frequency of SMS: 5.2%
10. Age Group: 2.9%
```
