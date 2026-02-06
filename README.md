# 🚀 An Intelligent End-to-End Customer Churn Prediction System

---

## 📌 Project Overview

**ChurnSense** is a complete end-to-end machine learning project designed to predict
customer churn in the **telecom domain**.

The project follows an **intelligent, automated ML pipeline** where every decision —
from data preprocessing to model selection — is driven by **statistical validation
and performance metrics recorded in log files**, rather than manual assumptions.

---

## 🎯 Problem Statement

Customer churn is a major challenge in the telecom industry. Retaining existing
customers is far more cost-effective than acquiring new ones.

**Objective:**  
Predict whether a customer is likely to **churn (Yes / No)** using historical customer
data, enabling proactive customer retention strategies.

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
```
---
## 📊 Dataset Description

The dataset contains telecom customer information including:

- Customer demographics (gender, senior citizen, partner, dependents)
- Service usage details (internet service, phone service, streaming services)
- Contract type, payment method, and billing information
- Target variable: **Churn**


### 📐 Dataset Shape

- **Number of Rows:** `7043`
- **Number of Columns:** `24`

### Files Used
- `Churn.csv` → Raw dataset  
- `Churn_Updated.csv` → Cleaned dataset after preprocessing  

---


## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand customer behavior and data characteristics before
building machine learning models.

### EDA Activities
- Churn vs Non-Churn distribution analysis
- Distribution analysis of numerical features:
  - Tenure
  - MonthlyCharges
  - TotalCharges
- Missing value identification
- Visual outlier detection using plots
- Correlation analysis between features and churn

### Tools Used
- **Matplotlib**
- **Seaborn**

### Files
- `eda.py`
- `EDA_Task-1.pdf`
- Logs: `logs/eda.log`

---

# ⚙️ Intelligent Machine Learning Pipeline

Each stage of the pipeline evaluates **multiple techniques**, logs performance
statistics, and **automatically selects the best method**.

---

## 1️⃣ 🧩 Missing Value Handling

Missing values were identified primarily in the **TotalCharges** column.
Instead of applying a single imputation method blindly, **multiple techniques were
evaluated for the affected column**, and the best one was selected automatically
based on logged statistical similarity scores.

---

## 🔍 Columns Considered

- TotalCharges

---

## 🧪 Techniques Evaluated (Per Column)

For the column containing missing values, the following techniques were evaluated:

- Mean Imputation
- Median Imputation
- Mode Imputation
- Random Sample Imputation
- KNN Imputation
- Iterative Imputation
- Forward Fill
- Backward Fill
- Interpolation

---

## ⚙️ Selection Logic

For the **TotalCharges** column:

- Mean difference between original and imputed data was calculated
- Standard deviation difference was calculated
- A **combined score** was computed as:
- Score = |mean_diff| + |std_diff|
- The technique with the **lowest combined score** was selected automatically
- This ensured **maximum preservation of the original data distribution**

---

## 🏆 Selected Technique per Column (From Logs)

### 🔹 TotalCharges

| Technique | Mean Diff | Std Diff | Score |
|---------|----------|---------|-------|
| Mean | 0.000000 | 2.009726 | 2.311184 |
| Median | 1.562575 | 1.706045 | 3.758913 |
| Mode | 4.030803 | 0.010286 | 4.647253 |
| **Random Sample** | **1.221698** | **0.326336** | **1.315828** ✅ |
| KNN | 0.000000 | 2.009726 | 1.708267 |
| Iterative | 0.000000 | 2.009726 | 1.708267 |
| Forward Fill | 1.926153 | 0.868170 | 2.794323 |
| Backward Fill | 2.330377 | 0.849112 | 3.179489 |
| Interpolation | 2.128265 | 1.039667 | 3.167932 |

---

## ✅ Final Selection

- **Selected Technique:** **Random Sample Imputation**
- **Selected Column:** TotalCharges

### Reason for Selection
- Lowest combined statistical similarity score
- Preserved both mean and variance effectively
- Reduced bias compared to constant-value imputation
- Selection confirmed through execution logs

---

## 📂 Files
- `missing_value_techniques.py`
- Logs: `missing_value_techniques.log`, `missing_values.log`


## 2️⃣ 🔄 Variable Transformation

Numerical features were transformed **independently per column** to reduce skewness.

### Techniques Evaluated (Per Column)
- Log
- Square Root
- Cube Root
- Box-Cox
- Yeo-Johnson
- Quantile
- Reciprocal
- Exponential

### Selection Logic
- Skewness calculated before and after transformation
- Absolute skewness used as selection score
- Lowest skewness selected automatically

### Selected Transformation per Column

| Column | Selected Technique |
|------|-------------------|
| SeniorCitizen | Skipped (Binary) |
| tenure | Box-Cox |
| MonthlyCharges | Quantile |
| TotalCharges | Quantile |

### Files
- `variable_transformation.py`
- Logs: `variable_transformation.log`

## 3️⃣ 🚨 Outlier Detection & Treatment

Outliers were detected and treated using multiple statistical techniques.
The best method was **automatically selected per column** based on a
combined evaluation score.

### Techniques Evaluated
- IQR Capping
- Z-Score
- MAD (Median Absolute Deviation)
- Percentile Capping
- Winsorization
- Clipping

### 📏 Selection Logic

Each technique was evaluated using a **combined score**:
- Score = |Skewness| + |Kurtosis| + |Outlier Ratio|
- Lower score ⇒ better distribution after treatment
- Preserves shape while minimizing extreme values
- Binary / quasi-constant features were skipped

### 🧠 Column-wise Selected Techniques

| Column Name       | Selected Technique | Reason |
|-------------------|-------------------|--------|
| `SeniorCitizen`   | ❌ Skipped | Binary / quasi-constant feature |
| `tenure`          | ❌ No Treatment | No significant outliers detected |
| `MonthlyCharges` | ✅ **Z-Score** | Lowest combined score (best balance of skewness, kurtosis & outlier ratio) |
| `TotalCharges`   | ✅ **Z-Score** | Achieved minimum combined score with stable distribution |

### 🏆 Final Selection Summary
- **Z-Score method** consistently achieved the **lowest combined score**
- Preserved distribution symmetry
- Controlled extreme values without aggressive clipping

### Files
- `outlier_handling.py`
- Logs: `outlier_handling.log`
- Plots: `plot_outliers/`

---


## 4️⃣ 🔠 Categorical to Numerical Encoding

Categorical features were converted into numerical form using a **rule-based encoding strategy**
to ensure compatibility with machine learning models while controlling dimensionality.

### 🔧 Encoding Techniques Used

- **Label Encoding**
- **One-Hot Encoding**

### 🧠 Selection Logic

- **Binary categorical features (2 unique values)**  
  → Applied **Label Encoding**  
  - Preserves information without increasing feature space
  - Ideal for Yes/No type variables

- **Multi-class categorical features (> 2 unique values)**  
  → Applied **One-Hot Encoding**  
  - Prevents ordinal bias
  - Allows the model to treat each category independently

### 🏆 Final Encoding Strategy

- Features with **exactly two categories** were encoded using **Label Encoding**
- Features with **more than two categories** were encoded using **One-Hot Encoding**
- Encoding decisions were applied **automatically per column** based on unique value count

### Files
- `cat_to_num_techniques.py`
- Logs: `cat_to_num_techniques.log`

## 5️⃣ 🎯 Feature Selection

Feature selection was performed to retain only statistically relevant and informative
features while reducing noise and overfitting.

### 🔧 Techniques Used

1. **Constant Feature Removal**
2. **Quasi-Constant Feature Removal**
3. **Statistical Hypothesis Testing (Pearson Correlation)**

---

### 🧠 Selection Logic

#### 1️⃣ Constant Feature Removal
- Features with **zero variance** were removed
- These features carry no information for prediction

**Technique Used**
- `VarianceThreshold(threshold = 0.0)`

---

#### 2️⃣ Quasi-Constant Feature Removal
- Features with **very low variance** (dominant single value) were removed
- Helps eliminate near-constant noise features

**Technique Used**
- `VarianceThreshold(threshold = 0.1)`



#### 3️⃣ Hypothesis Testing (Pearson Correlation)
- Tested statistical relationship between each feature and the target variable (**Churn**)
- Pearson correlation p-value computed for each feature

**Decision Rule**
```text
If p-value > 0.05 → Feature removed
If p-value ≤ 0.05 → Feature retained
```
---

## 6️⃣ ⚖️ Data Balancing

Class imbalance between churned and non-churned customers was handled using a
**conditional, decision-driven balancing strategy**.

Instead of blindly applying oversampling, the pipeline **first evaluates the imbalance
severity** and only applies balancing when necessary.


### 🎯 Purpose
- Improve learning of the minority (churned) class
- Prevent model bias toward the majority class
- Avoid unnecessary noise caused by over-oversampling


### 🔍 Class Distribution (Before Balancing)

- Class 0 (No Churn): **4138**
- Class 1 (Churn): **1496**

**Imbalance Ratio**
```text
Imbalance Ratio = max(class_count) / min(class_count)
               = 4138 / 1496 ≈ 2.76
```
---

## 7️⃣ 📐 Feature Scaling

Feature scaling was applied to normalize numerical features and improve model
stability and convergence.

Multiple scaling techniques were evaluated, and the **best scaler was selected
automatically using statistical scoring**, instead of manual choice.

### 🔧 Scalers Evaluated
- StandardScaler
- MinMaxScaler
- RobustScaler
- MaxAbsScaler

### 🧠 Selection Logic

- Each scaler was applied independently on the training data
- A **statistical score** was computed for every scaled dataset
- The scaler with the **best (lowest) score** was selected
- This ensures:
  - Reduced influence of outliers
  - Stable feature distributions
  - Better downstream model performance

> ⚠️ Note: All scalers achieved identical scores in this dataset.  
> In such cases, a **robust default** was chosen.

### 🏆 Selected Scaler

✅ **RobustScaler**

**Reason**
- Performs well in the presence of outliers
- Uses median and interquartile range (IQR)
- Preferred when multiple scalers show equal performance

### 📤 Output
- Best scaler saved as:  
  **`scaler_path.pkl`**

This scaler is reused during:
- Model training
- Inference
- Web application deployment

---

## 8️⃣ 🤖 Model Training & Evaluation

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

## 🏆 Best Model Selection & Tuning

### ROC-AUC Scores (From Logs)
- **Naive Bayes: 0.7719 (Best)**
- AdaBoost: 0.7466
- Gradient Boosting: 0.7410
- Logistic Regression: 0.7316

### Selected Model
✅ **Naive Bayes**

### Hyperparameter Tuning
- GridSearchCV applied
- Best parameter: `var_smoothing = 1e-12`

### Saved Model
- `Churn_Prediction_Best_Model.pkl`

---

## 🌐 Model Deployment (Flask)

The final tuned model was deployed as a web application.

### Features
- User input form
- Churn prediction (Yes / No)
- Churn probability output

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

## ⚙️ Installation & Setup

```bash
pip install -r requirements.txt
python app.py
```
---

## 👩‍💻 Author

**Sanjana Katta**  
B.Tech Student | Computer Science & Engineering  
Machine Learning & Data Science Enthusiast  
Email - sanjanaa.katta20@gmail.com

📌 Focus Areas:
- End-to-End Machine Learning Pipelines  
- Data Preprocessing & Feature Engineering  
- Model Evaluation & Deployment  
