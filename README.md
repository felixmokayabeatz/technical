# Predictive Analytics Using Machine Learning

## 1. Dataset Description

For this project, we used a *Customer Churn Dataset* to predict whether a customer will churn based on various attributes such as tenure, monthly charges, total charges, internet service type, contract type, and payment method.

The dataset consists of:

- **Features:** Tenure, monthly charges, total charges, contract type, internet service type, etc.
- **Target Variable:** Churn (Yes/No)

## 2. Exploratory Data Analysis (EDA)

### 2.1 Data Loading & Overview

- The dataset was loaded into a Pandas DataFrame and checked for missing values.
- Summary statistics were computed to understand the distribution of numerical features.

### 2.2 Handling Missing Values

- Missing values in numerical columns were imputed with the median.
- Missing values in categorical columns were filled with the mode.

### 2.3 Outlier Detection & Handling

- Box plots and histograms were used to detect outliers.
- Outliers in numerical columns were capped at the 1st and 99th percentiles.

### 2.4 Encoding Categorical Variables

- One-hot encoding was applied to categorical features such as contract type and internet service type.

### 2.5 Feature Scaling

- Numerical features were standardized using **StandardScaler** to ensure uniform distribution.

## 3. Feature Engineering

- Feature selection was performed using **correlation analysis** and **feature importance techniques**.
- The most important features identified were:
  - Tenure
  - Monthly Charges
  - Contract Type
- New features such as **total monthly spend ratio** were created to enhance model performance.

## 4. Model Training & Evaluation

### 4.1 Model Selection

We trained the following models:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **XGBoost Classifier**

### 4.2 Model Evaluation Metrics

The models were evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

Model    Accuracy  Precision  Recall  F1-score

Logistic Regression  80.2%  78.5%  76.3%  77.4%

Decision Tree  76.4%  74.2%  72.8%  73.5%

Random Forest  85.1%  83.7%  81.9%  82.8%

XGBoost  86.3%   85.1%  84.0%  84.5%

- The **XGBoost model** had the highest F1-score and was selected as the best model.

### 4.3 Hyperparameter Tuning

- Hyperparameters were optimized using **GridSearchCV**.
- The best hyperparameters for XGBoost:
  - `n_estimators = 100`
  - `max_depth = 5`
  - `learning_rate = 0.1`

## 5. Feature Importance Analysis

- The most important features identified by the XGBoost model were:
  - **Contract Type** (Month-to-month contracts had higher churn rates)
  - **Tenure** (Longer tenure customers were less likely to churn)
  - **Monthly Charges** (Higher charges correlated with churn)

## 6. Model Deployment (Bonus)

- The trained model was saved using `joblib`.
- A **Flask application** was created to serve predictions.

## 7. Final Insights & Recommendations

### 7.1 Key Findings

- Customers with **month-to-month contracts** had the highest churn rate.
- **Higher monthly charges** increased churn likelihood.
- **Longer tenure customers** were more loyal.

### 7.2 Recommendations for the Business

- Offer **discounted long-term contracts** to reduce churn.
- Provide **better customer support** to high-risk customers.
- Implement **loyalty programs** to retain long-term customers.

## 8. References

- Dataset: Kaggle – Customer Churn Prediction Dataset
- Machine Learning Libraries: Scikit-learn, XGBoost, Pandas, NumPy

---
**Author:** Felix Mokaya  
**Institution:** Karatina University  
**Date:** March 2025
