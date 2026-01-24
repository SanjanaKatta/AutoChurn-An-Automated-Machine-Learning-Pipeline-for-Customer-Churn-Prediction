# 🚀 ChurnSense: An Intelligent End-to-End Customer Churn Prediction System

ChurnSense is a complete **end-to-end machine learning project** designed to predict **customer churn in the telecom domain**. The project covers the entire ML lifecycle — from **exploratory data analysis (EDA)** and **data preprocessing**, to **automated model selection**, **performance evaluation**, and **deployment using Flask**.

The key objective of this project is to build an **intelligent and automated pipeline** that selects the best preprocessing techniques and machine learning model based on statistical and performance-driven criteria, rather than relying on manual assumptions.

---

## 📌 Problem Statement

Customer churn is a major challenge in the telecom industry, as acquiring new customers is significantly more expensive than retaining existing ones. By predicting churn in advance, telecom companies can take proactive retention actions such as personalized offers, service improvements, and targeted communication.

This project aims to accurately predict whether a customer is likely to churn based on historical customer data.

---

## 📊 Dataset Description

The dataset contains telecom customer information including:
- Customer demographics (gender, senior citizen, dependents, partner)
- Service usage details (internet service, phone service, streaming, security, tech support)
- Contract type, payment method, and billing information
- Target variable **Churn** indicating whether the customer left the service

Files used:
- `Churn.csv` → Raw dataset  
- `Churn_Updated.csv` → Cleaned dataset after preprocessing  

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand customer behavior and data characteristics. Both **script-based and notebook-based EDA** were implemented.

### Key EDA activities:
- Churn vs non-churn distribution analysis
- Numerical feature distribution analysis (tenure, monthly charges, total charges)
- Detection of missing values and outliers
- Correlation analysis between features and churn

### EDA files:
- `EDA.ipynb`
- `eda.py`
- `EDA_Task-1.pdf`
- `EDA_Task-1.docx`
- Logs: `logs/eda.log`

---

## ⚙️ Intelligent Machine Learning Pipeline

The heart of this project is an **automated ML pipeline** implemented in `main.py`, where multiple preprocessing techniques are evaluated and the best ones are selected automatically.

---

### 🧩 Missing Value Handling
Multiple techniques were applied and evaluated:
- Mean, Median, Mode
- Random Imputation
- KNN Imputation
- Iterative Imputation  

The best technique was selected based on **distribution similarity (mean & standard deviation preservation)**.

Files:
- `missing_value_techniques.py`
- Logs: `missing_value_techniques.log`, `missing_values.log`

---

### 🔄 Variable Transformation
To handle skewness in numerical features, multiple transformations were tested:
- Log
- Square Root
- Cube Root
- Box-Cox
- Yeo-Johnson
- Quantile Transformation  

The transformation with **minimum skewness** was automatically selected.

Files:
- `variable_transformation.py`
- Logs: `variable_transformation.log`

---

### 🚨 Outlier Detection & Treatment
Several outlier-handling techniques were evaluated:
- IQR Capping
- Z-Score
- MAD
- Percentile Capping
- Winsorization
- Clipping  

The best technique was selected using a **combined score of skewness, kurtosis, and remaining outlier ratio**.

Files:
- `outlier_handling.py`
- Logs: `outlier_handling.log`
- Folder: `plot_outliers/`

---

### 🔠 Categorical to Numerical Encoding
Categorical features were handled intelligently:
- Binary features → Label Encoding
- Multi-class features → One-Hot / Frequency / Binary Encoding (evaluated automatically)

Files:
- `cat_to_num_techniques.py`
- Logs: `cat_to_num_techniques.log`

---

### 🎯 Feature Selection
Feature selection techniques were applied to retain only the most relevant features, improving model performance and reducing complexity.

Files:
- `feature_selection.py`
- Logs: `feature_selection.log`

---

### ⚖️ Data Balancing
Class imbalance was handled using balancing techniques to ensure better learning for churned customers.

Files:
- `data_balancing.py`
- Logs: `data_balancing.log`

---

### 📐 Feature Scaling
Multiple scalers were evaluated:
- StandardScaler
- MinMaxScaler
- RobustScaler
- MaxAbsScaler  

The best-performing scaler was selected and saved.

File:
- `scaler_path.pkl`

---

## 🤖 Model Training & Evaluation

Multiple machine learning models were trained and evaluated using **ROC-AUC score** as the primary metric.

Models trained:
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- Support Vector Machine (SVM)

Files:
- `All_models.py`
- Logs: `All_Models.log`
- Performance plot: `ROC_AUC_Curve.png`

---

## 🏆 Best Model Selection

Based on ROC-AUC comparison, **Naive Bayes** was selected as the best-performing model. The trained model was saved for deployment.

File:
- `Churn_Prediction_Best_Model.pkl`

---

## 🌐 Model Deployment (Flask)

The final model was deployed using a **Flask web application** that allows users to input customer details and receive:
- Churn prediction (Yes / No)
- Churn probability

Frontend technologies:
- HTML (`index.html`)
- CSS (`style.css`)
- JavaScript (`script.js`)

File:
- `app.py`

---


## 🛠 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Flask
- HTML, CSS, JavaScript
- Matplotlib, Seaborn

---

## 📁 Project Structure

```text
├── app.py                          # Flask application for deployment
├── main.py                         # End-to-end automated ML pipeline
├── All_models.py                   # Model training, evaluation & selection
│
├── templates/
│   └── index.html                  # Frontend UI
│
├── static/
│   ├── style.css                   # UI styling
│   └── script.js                   # Frontend logic
│
├── Churn.csv                       # Raw dataset
├── Churn_Updated.csv               # Preprocessed dataset
│
├── Churn_Prediction_Best_Model.pkl # Saved best ML model
├── scaler_path.pkl                 # Saved feature scaler
│
├── logs/                           # Execution & training logs
├── EDA.ipynb                       # Exploratory Data Analysis
├── ROC_AUC_Curve.png               # Model performance visualization
└── README.md
