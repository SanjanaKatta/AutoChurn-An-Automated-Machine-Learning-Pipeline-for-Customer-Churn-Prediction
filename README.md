 # 🚀An Intelligent End-to-End Customer Churn Prediction System

---

## 📌 Project Overview

**ChurnSense** is a complete end-to-end machine learning project designed to predict
customer churn in the **telecom domain**.

The project follows an **intelligent, automated ML pipeline** where every decision —
from data preprocessing to model selection — is made using **statistical validation
and performance metrics recorded in log files**, instead of manual assumptions.

---

## 🎯 Problem Statement

Customer churn is a major challenge in the telecom industry. Retaining existing
customers is far more cost-effective than acquiring new ones.

**Objective:**  
Predict whether a customer is likely to **churn (Yes / No)** using historical customer
data, enabling proactive customer retention strategies.

---

## 📊 Dataset Description

The dataset contains telecom customer information including:

- Customer demographics (gender, senior citizen, partner, dependents)
- Service usage details (internet service, phone service, streaming services)
- Contract type, payment method, and billing information
- Target variable: **Churn**

### Files Used
- `Churn.csv` → Raw dataset  
- `Churn_Updated.csv` → Cleaned dataset after preprocessing  

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand customer behavior, data distribution, and potential
issues before applying machine learning models.

### EDA Activities Performed
- Churn vs Non-Churn distribution analysis
- Numerical feature distribution analysis:
  - Tenure
  - MonthlyCharges
  - TotalCharges
- Missing value detection
- Outlier detection using visual analysis
- Correlation analysis between features and churn

### Purpose of EDA
- Identify data quality issues
- Understand churn-driving patterns
- Guide preprocessing decisions

### Files
- `eda.py`
- `EDA_Task-1.pdf`
- Logs: `logs/eda.log`

---

## ⚙️ Intelligent Machine Learning Pipeline

The pipeline is implemented in `main.py` and follows a **step-by-step automated
selection strategy**, where multiple techniques are evaluated and the **best one is
chosen using logged statistical and performance metrics**.

---

## 🧩 Missing Value Handling

Missing values were detected primarily in the **TotalCharges** column.

### Techniques Evaluated
- Mean Imputation  
- Median Imputation  
- Mode Imputation  
- Random Sample Imputation  
- KNN Imputation  
- Iterative Imputation  

### Selection Logic (Per Technique)
- **Mean / Median / Mode:** Compared impact on mean and standard deviation
- **Random Sample:** Checked preservation of original distribution
- **KNN / Iterative:** Evaluated variance distortion and computation cost

### Selected Technique
✅ **Random Sample Imputation**

### Reason for Selection
- Best preservation of original data distribution
- Minimal change in mean and standard deviation
- Reduced bias compared to constant-value imputation
- Confirmed through logged statistical similarity scores

### Files
- `missing_value_techniques.py`
- Logs: `missing_value_techniques.log`, `missing_values.log`

---

## 🔄 Variable Transformation

To reduce skewness and normalize numerical features, multiple transformations were
evaluated automatically.

### Techniques Evaluated & Selection Logic

- **Log Transformation**
- **Square Root Transformation**
- **Cube Root Transformation**
- **Box-Cox Transformation**
- **Yeo-Johnson Transformation**
- **Quantile Transformation**

### Selection Logic
- Skewness calculated **before and after** each transformation
- Transformation with **minimum resulting skewness** was selected automatically

### Selected Technique
✅ **Yeo-Johnson Transformation**

### Reason for Selection
- Achieved lowest skewness across numerical features
- Works with zero and negative values
- Preserved feature relationships better than quantile-based methods

### Files
- `variable_transformation.py`
- Logs: `variable_transformation.log`

---

## 🚨 Outlier Detection & Treatment

Outliers were handled using multiple statistical techniques.

### Techniques Evaluated
- IQR Capping  
- Z-Score  
- MAD  
- Percentile Capping  
- Winsorization  
- Clipping  

### Selection Logic
Each technique was scored using:
- Post-treatment skewness
- Kurtosis
- Remaining outlier ratio

### Selected Technique
✅ **Best technique selected automatically based on combined score**

### Files
- `outlier_handling.py`
- Logs: `outlier_handling.log`
- Folder: `plot_outliers/`

---

## 🔠 Categorical to Numerical Encoding

Categorical variables were converted to numerical form intelligently.

### Encoding Strategy
- Binary categorical features → Label Encoding
- Multi-class categorical features → Evaluated among:
  - One-Hot Encoding
  - Frequency Encoding
  - Binary Encoding

### Selection Logic
- Based on model performance and dimensionality impact

### Files
- `cat_to_num_techniques.py`
- Logs: `cat_to_num_techniques.log`

---

## 🎯 Feature Selection

Feature selection techniques were applied to retain only the most relevant features.

### Selection Logic
- Removed low-importance and redundant features
- Improved model generalization
- Reduced overfitting risk

### Files
- `feature_selection.py`
- Logs: `feature_selection.log`

---

## ⚖️ Data Balancing

Class imbalance between churned and non-churned customers was addressed.

### Selection Logic
- Ensured better learning for minority (churned) class
- Prevented model bias toward majority class

### Files
- `data_balancing.py`
- Logs: `data_balancing.log`

---

## 📐 Feature Scaling

Multiple scaling techniques were evaluated.

### Scalers Evaluated
- StandardScaler  
- MinMaxScaler  
- RobustScaler  
- MaxAbsScaler  

### Selection Logic
- Based on downstream model ROC-AUC performance

### Selected Scaler
✅ **Best-performing scaler selected and saved**

### File
- `scaler_path.pkl`

---

## 🤖 Model Training & Evaluation

### Models Trained
- KNN  
- Naive Bayes  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- AdaBoost  
- Gradient Boosting  
- XGBoost  
- SVM  

### Evaluation Metrics
- Accuracy
- **ROC-AUC (Primary Metric)**

### Files
- `All_models.py`
- Logs: `All_Models.log`
- ROC Curve: `ROC_AUC_Curve.png`

---

## 🏆 Best Model Selection

### ROC-AUC Scores (from logs)
- KNN: 0.6385  
- **Naive Bayes: 0.7719**  
- Logistic Regression: 0.7316  
- Decision Tree: 0.6360  
- Random Forest: 0.6631  
- AdaBoost: 0.7466  
- Gradient Boosting: 0.7410  
- XGBoost: 0.6898  
- SVM: 0.4975  

### Selected Best Model
✅ **Naive Bayes**

### Reason
- Highest ROC-AUC score
- Stable probabilistic predictions
- Best performance after automated preprocessing

### Hyperparameter Tuning
- Tuned using GridSearchCV
- Best parameter: `var_smoothing = 1e-12`

### Saved Model
- `Churn_Prediction_Best_Model.pkl`

---

## 🌐 Model Deployment (Flask)

The final tuned model was deployed using Flask.

### Features
- User inputs customer details
- Predicts churn (Yes / No)
- Displays churn probability

### Files
- `app.py`
- `templates/index.html`
- `static/style.css`
- `static/script.js`

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

## ✅ Conclusion

ChurnSense demonstrates a **fully automated, production-ready ML system** where
every preprocessing and modeling decision is driven by **logged statistical evidence
and performance metrics**, ensuring transparency, reliability, and real-world
applicability.

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
