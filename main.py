'''
In this file we are going to load the data and other ML pipeline techniques which are needed
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('main')

from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from scipy import stats
from sklearn.impute import SimpleImputer

from missing_value_techniques import MISSING_VALUES
from variable_transformation import VARIABLE_TRANSFORMATION
from outlier_handling import OUTLIER_HANDLING
from feature_selection import FEATURE_SELECTION
from cat_to_num_techniques import CAT_TO_NUM_TECHNIQUES
from data_balancing import DATA_BALANCING
from All_models import common

class CHURN_PREDICTION:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)

            logger.info('Data loaded successfully')

            logger.info(f'Total rows in the data : {self.df.shape[0]}')
            logger.info(f'Total columns in the data : {self.df.shape[1]}')
            logger.info(f'{self.df.tail(5)}')

            self.y = self.df['Churn'].map({'Yes': 1, 'No': 0})
            self.X = self.df.drop(columns=['Churn', 'customerID'])

            logger.info(f'Independent Column(X): {self.X.shape}')
            logger.info(f'Dependent Column(y): {self.y.shape}')

            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 42)

            logger.info(f'X_train Columns:{self.X_train.columns}')
            logger.info(f'X_test Columns:{self.X_test.columns}')

            logger.info(f'y_train Sample Data: {self.y_train.sample(5)}')
            logger.info(f'y_test Sample Data: {self.y_test.sample(5)}')

            logger.info(f'Training data size : {self.X_train.shape}')
            logger.info(f'Testing data size : {self.X_test.shape}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def missing_values(self):
        try:
            logger.info('Selecting Missing Value Technique')

            techniques = {
                'mean': MISSING_VALUES.mean_imputation,
                'median': MISSING_VALUES.median_imputation,
                'mode': MISSING_VALUES.mode_imputation,
                'random': MISSING_VALUES.random_imputation,
                'knn': MISSING_VALUES.knn_imputation,
                'iterative': MISSING_VALUES.iterative_imputation,
                'ffill': MISSING_VALUES.forward_fill,
                'bfill': MISSING_VALUES.backward_fill,
                'interpolation': MISSING_VALUES.interpolation
            }

            X_train_filled = self.X_train.copy()
            X_test_filled = self.X_test.copy()

            missing_cols = self.X_train.columns[self.X_train.isnull().any()]

            for col in missing_cols:
                logger.info(f'Evaluating techniques for column: {col}')
                scores = {}

                base = self.X_train[col].dropna()
                base_mean = base.mean()
                base_std = base.std()

                for name, func in techniques.items():
                    X_tr_col, _ = func(self.X_train[[col]].copy(),self.X_test[[col]].copy())

                    imputed = X_tr_col[col]

                    mean_diff = abs(imputed.mean() - base_mean)
                    std_diff = abs(imputed.std() - base_std)

                    score = mean_diff + std_diff

                    #PROJECT LOGIC BOOST
                    # Prefer stochastic methods if distribution is preserved
                    if name in ['random', 'iterative', 'knn']:
                        score *= 0.85  # reward variance preservation

                    # Penalize deterministic methods
                    if name in ['mean', 'median', 'mode']:
                        score *= 1.15

                    scores[name] = score

                    logger.info(f'{name:<15} -> Score: {score:.6f} 'f'(mean_diff={mean_diff:.6f}, std_diff={std_diff:.6f})')

                #FORCE RANDOM IF CLOSE ENOUGH
                best_score = min(scores.values())
                close_methods = [
                    k for k, v in scores.items()
                    if abs(v - best_score) < 0.05]

                if 'random' in close_methods:
                    best_tech = 'random'
                else:
                    best_tech = min(scores, key=scores.get)

                X_train_filled[col], X_test_filled[col] = techniques[best_tech](self.X_train[[col]].copy(),self.X_test[[col]].copy())

                logger.info(f'BEST technique for {col}: {best_tech.upper()} 'f'| Score={scores[best_tech]:.6f}')

            self.X_train = X_train_filled
            self.X_test = X_test_filled

            logger.info('Missing Value Handling Completed')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in Line {error_line.tb_lineno} : {error_msg}')

    def vt_out(self):
        try:
            #logger.info(f'{self.X_train.info()}')
            logger.info('Selecting Variable Transformation Technique')
            logger.info(f'X_train Columns : {self.X_train.columns}')
            logger.info(f'X_test Columns: {self.X_test.columns}')

            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')

            logger.info(f'X_train Numerical columns : {self.X_train_num.columns}')
            logger.info(f'X_train Categorical columns: {self.X_train_cat.columns}')
            logger.info(f'X_test Numerical columns: {self.X_test_num.columns}')
            logger.info(f'X_test Categorical columns: {self.X_test_cat.columns}')

            logger.info(f'X_train Numerical Shape: {self.X_train_num.shape}')
            logger.info(f'X_train Categorical Shape: {self.X_train_cat.shape}')
            logger.info(f'X_test Numerical Shape: {self.X_test_num.shape}')
            logger.info(f'X_test Categorical Shape: {self.X_test_cat.shape}')

            X_train_transformed = self.X_train_num.copy()
            X_test_transformed = self.X_test_num.copy()

            techniques = {
                'log': VARIABLE_TRANSFORMATION.log_transform,
                'boxcox': VARIABLE_TRANSFORMATION.boxcox_transform,
                'yeojohnson': VARIABLE_TRANSFORMATION.yeojohnson_transform,
                'quantile': VARIABLE_TRANSFORMATION.quantile_transform,
                'reciprocal': VARIABLE_TRANSFORMATION.reciprocal_transform,
                'sqrt': VARIABLE_TRANSFORMATION.squareroot_transform,
                'cuberoot': VARIABLE_TRANSFORMATION.cuberoot_transform,
                'exp': VARIABLE_TRANSFORMATION.exponential_transform
            }

            for col in self.X_train_num.columns:
                logger.info(f'Evaluating transformations for column: {col}')

                if self.X_train_num[col].nunique() <= 2:
                    logger.info(f'Skipping {col} (binary/discrete feature)')
                    X_train_transformed[col] = self.X_train_num[col]
                    X_test_transformed[col] = self.X_test_num[col]
                    continue

                scores = {}

                for name, func in techniques.items():
                    X_tr_col, _ = func(self.X_train_num[[col]].copy(),self.X_test_num[[col]].copy())
                    skew_val = skew(X_tr_col[col], nan_policy='omit')
                    score = abs(skew_val)
                    scores[name] = score
                    logger.info(f'{name:<15} -> Skew: {skew_val:.4f}, 'f'Score: {score:.4f}')

                # Select best technique
                best_tech = min(scores, key=scores.get)

                # Apply best transformation
                X_tr_best, X_te_best = techniques[best_tech](self.X_train_num[[col]].copy(),self.X_test_num[[col]].copy())

                X_train_transformed[col] = X_tr_best[col]
                X_test_transformed[col] = X_te_best[col]

                logger.info(f'BEST transformation for {col}: {best_tech} | 'f'Score: {scores[best_tech]:.4f}')

            self.X_train_num = X_train_transformed
            self.X_test_num = X_test_transformed

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def outlier(self):
        try:
            logger.info('Selecting Best Outlier Technique (Skew + Kurtosis + Outlier Ratio)')

            # Apply all techniques
            all_outlier_results = OUTLIER_HANDLING.apply_all_techniques(
                X_train=self.X_train_num,
                X_test=self.X_test_num
            )

            techniques = {
                'iqr': OUTLIER_HANDLING.iqr_capping,
                'zscore': OUTLIER_HANDLING.zscore_capping,
                'mad': OUTLIER_HANDLING.mad_capping,
                'percentile': OUTLIER_HANDLING.percentile_capping,
                'winsor': OUTLIER_HANDLING.winsorization,
                'clip': OUTLIER_HANDLING.clipping
            }

            X_train_out = self.X_train_num.copy()
            X_test_out = self.X_test_num.copy()

            for col in self.X_train_num.columns:

                # Skip binary/quasi-constant features
                if self.X_train_num[col].nunique() <= 2:
                    logger.info(f"{col} is binary/quasi-constant --> skipping outlier detection")
                    continue

                # IQR quick check
                Q1 = self.X_train_num[col].quantile(0.25)
                Q3 = self.X_train_num[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

                if not ((self.X_train_num[col] < lower) | (self.X_train_num[col] > upper)).any():
                    logger.info(f'No outliers detected for column: {col}')
                    X_train_out[col] = self.X_train_num[col]
                    X_test_out[col] = self.X_test_num[col]
                    continue

                scores = {}

                for name in techniques.keys():
                    X_tr_col, _ = all_outlier_results[col][name]
                    data = X_tr_col[col].dropna()

                    # ---- Metrics ----
                    sk = abs(skew(data))
                    kt = abs(kurtosis(data, fisher=True))

                    # Outlier ratio after technique
                    Q1_t = data.quantile(0.25)
                    Q3_t = data.quantile(0.75)
                    IQR_t = Q3_t - Q1_t
                    lower_t, upper_t = Q1_t - 1.5 * IQR_t, Q3_t + 1.5 * IQR_t
                    outlier_ratio = ((data < lower_t) | (data > upper_t)).mean()

                    score = sk + kt + outlier_ratio
                    scores[name] = score

                    logger.info(
                        f'{col} | {name:<18} '
                        f'Skew: {sk:.4f} | Kurtosis: {kt:.4f} | '
                        f'Outlier Ratio: {outlier_ratio:.4f} | Score: {score:.4f}'
                    )

                best_tech = min(scores, key=scores.get)

                X_tr_best, X_te_best = all_outlier_results[col][best_tech]
                X_train_out[col] = X_tr_best[col].values
                X_test_out[col] = X_te_best[col].values

                logger.info(
                    f'BEST outlier method for {col}: {best_tech} '
                    f'(Score: {scores[best_tech]:.4f})'
                )

            # Update numerical features
            self.X_train_num = X_train_out
            self.X_test_num = X_test_out

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no: {error_line.tb_lineno} due to {error_msg}')

    def cat_to_num(self):
        try:
            logger.info("===================================================")
            logger.info("Categorical to Numerical Encoding Started")
            logger.info("===================================================")

            logger.info(f"X_train_cat shape: {self.X_train_cat.shape}")
            logger.info(f"X_test_cat shape : {self.X_test_cat.shape}")

            # If no categorical columns
            if self.X_train_cat.empty:
                logger.info("No categorical columns found. Skipping encoding.")
                return

            # Output encoded data
            self.X_train_enc = pd.DataFrame(index=self.X_train_cat.index)
            self.X_test_enc = pd.DataFrame(index=self.X_test_cat.index)

            for col in self.X_train_cat.columns:

                unique_vals = self.X_train_cat[col].dropna().nunique()
                logger.info(f"Processing Column: {col} | Unique Values: {unique_vals}")

                # ------------------ Binary Feature ------------------
                if unique_vals == 2:
                    logger.info(f"{col} --> Binary Feature | Applying Label Encoding")

                    X_tr_col, X_te_col = CAT_TO_NUM_TECHNIQUES.label_encoding(
                        self.X_train_cat[[col]],
                        self.X_test_cat[[col]]
                    )

                # ------------------ Multi-class Feature ------------------
                elif unique_vals > 2:
                    logger.info(f"{col} --> Multi-class Feature | Applying One-Hot Encoding")

                    X_tr_col, X_te_col = CAT_TO_NUM_TECHNIQUES.one_hot_encoding(self.X_train_cat[[col]],self.X_test_cat[[col]])

                # ------------------ Useless Column ------------------
                else:
                    logger.info(f"{col} has ≤1 unique value --> Skipping")
                    continue

                # Prefix to avoid column collision
                X_tr_col = X_tr_col.add_prefix(f"{col}_")
                X_te_col = X_te_col.add_prefix(f"{col}_")

                # Merge encoded columns
                self.X_train_enc = pd.concat([self.X_train_enc, X_tr_col], axis=1)
                self.X_test_enc = pd.concat([self.X_test_enc, X_te_col], axis=1)

            logger.info("===================================================")
            logger.info("Encoding Completed Successfully")
            logger.info("===================================================")
            logger.info(f"Encoded Train Shape: {self.X_train_enc.shape}")
            logger.info(f"Encoded Test Shape : {self.X_test_enc.shape}")
            logger.info(f"Encoded Columns    : {list(self.X_train_enc.columns)}")

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in Line no : {error_line.tb_lineno} : due to {error_msg}")

    def f_selection(self):
        try:
            logger.info(f'Feature Selection started')

            logger.info(f"Before X_train Shape : {self.X_train_num.shape}")
            logger.info(f"Before X_test Shape  : {self.X_test_num.shape}")

            self.X_train_fs_num, self.X_test_fs_num = FEATURE_SELECTION.feature_selection(self.X_train_num,self.X_test_num,self.y_train)

            self.X_train_fs_num = self.X_train_fs_num.reset_index(drop=True)
            self.X_train_enc = self.X_train_enc.reset_index(drop=True)
            self.y_train = self.y_train.reset_index(drop=True)

            self.X_test_fs_num = self.X_test_fs_num.reset_index(drop=True)
            self.X_test_enc = self.X_test_enc.reset_index(drop=True)
            self.y_test = self.y_test.reset_index(drop=True)

            # SAFE CONCAT
            self.X_train = pd.concat([self.X_train_fs_num, self.X_train_enc],axis=1)
            self.X_test = pd.concat([self.X_test_fs_num, self.X_test_enc],axis=1)

            logger.info(f"After Feature Selection Train Shape : {self.X_train_fs_num.shape}")
            logger.info(f"After Feature Selection Test Shape  : {self.X_test_fs_num.shape}")

            logger.info(f'Training Data : {self.X_train.shape} --> {self.X_train.columns}')
            logger.info(f'y_train : {self.y_train.shape}')
            logger.info(f'Testing Data : {self.X_test.shape} --> {self.X_test.columns}')
            logger.info(f'y_test : {self.y_test.shape}')

            logger.info("Feature Selection Completed Successfully")

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def data_balance(self):
        try:
            logger.info('Data Balancing Technique')

            # 🔥 CRITICAL FIX: align X and y
            self.X_train = self.X_train.loc[self.y_train.index]
            self.X_test = self.X_test.loc[self.y_test.index]

            assert len(self.X_train) == len(self.y_train), "X_train & y_train mismatch"
            assert len(self.X_test) == len(self.y_test), "X_test & y_test mismatch"

            logger.info(f"Before X_train Shape : {self.X_train.shape}")
            logger.info(f'Before y_train shape : {self.y_train.shape}')
            logger.info(f"Before X_test Shape  : {self.X_test.shape}")
            logger.info(f'Before y_test shape : {self.y_test.shape}')

            logger.info(f'Before :No.of rows for Yes class : {sum(self.y_train == 1)}')
            logger.info(f'Before :No.of rows for No class : {sum(self.y_train == 0)}')


            X_train_bal, y_train_bal, X_test_bal = DATA_BALANCING.data_balancing(self.X_train,self.y_train,self.X_test,force_balance=True )

            self.X_train_bal = X_train_bal
            self.y_train_bal = y_train_bal
            self.X_test_bal = X_test_bal

            logger.info(f"After X_train Shape : {self.X_train_bal.shape}")
            logger.info(f"After X_test Shape : {self.X_test_bal.shape}")
            logger.info(f"After y_train Shape : {self.y_train_bal.shape}")
            logger.info(f"After y_test Shape : {self.y_test.shape}")

            logger.info(f'After :No.of rows for Yes class : {sum(self.y_train_bal == 1)}')
            logger.info(f'After :No.of rows for No class : {sum(self.y_train_bal == 0)}')

            logger.info('Data balancing Completed')

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def feature_scaling(self):
        try:
            logger.info("Selecting Best Feature Scaling Technique (Statistical Scoring)")

            #Use BALANCED data
            X_train = self.X_train_bal.copy()
            X_test = self.X_test_bal.copy()

            numeric_cols = X_train.columns.tolist()

            #Handle missing values FIRST
            imputer = SimpleImputer(strategy="median")

            X_train = pd.DataFrame(imputer.fit_transform(X_train),columns=numeric_cols,index=X_train.index)
            X_test = pd.DataFrame(imputer.transform(X_test),columns=numeric_cols,index=X_test.index)

            logger.info("Missing values handled using Median Imputation")

            # ===============================
            # 3. Scoring function (Skew + Kurtosis)
            # ===============================
            def score_scaled_df(df):
                sk = df.skew().abs().mean()
                kt = df.kurtosis().abs().mean()
                return sk + kt

            #scalers
            scalers = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler(),
                "maxabs": MaxAbsScaler()
            }

            best_score = np.inf
            best_scaler = None
            best_name = None

            #Evaluate each scaler
            for name, scaler in scalers.items():
                X_temp = X_train.copy()

                X_temp[numeric_cols] = scaler.fit_transform(X_temp[numeric_cols])

                score = score_scaled_df(X_temp[numeric_cols])
                logger.info(f"{name.upper()} scaler score = {score:.4f}")

                if score < best_score:
                    best_score = score
                    best_scaler = scaler
                    best_name = name

            logger.info(f"BEST SCALER SELECTED = {best_name.upper()}")

            #Apply BEST scaler
            self.X_train_bal[numeric_cols] = best_scaler.fit_transform(X_train[numeric_cols])
            self.X_test_bal[numeric_cols] = best_scaler.transform(X_test[numeric_cols])

            #Save scaler
            with open("scaler_path.pkl", "wb") as f:
                pickle.dump(best_scaler, f)

            #Logging & Safety checks
            logger.info("Feature stats AFTER scaling:\n" +self.X_train_bal[numeric_cols].describe().to_string())

            logger.info("X_train null values after scaling:\n" +self.X_train_bal.isnull().sum().to_string())

            logger.info("X_test null values after scaling:\n" +self.X_test_bal.isnull().sum().to_string())

            logger.info("Feature Scaling Completed Successfully")

            #calling common
            common(self.X_train_bal,self.y_train_bal,self.X_test_bal,self.y_test)

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in Line no : {error_line.tb_lineno} : due to {error_msg}")

if __name__ == '__main__':
    try:
        obj = CHURN_PREDICTION('C:\\Users\\Rajesh\\Downloads\\Churn Prediction\\Churn_Updated.csv')
        obj.missing_values()
        obj.vt_out()
        obj.cat_to_num()
        obj.outlier()
        obj.f_selection()
        obj.data_balance()
        obj.feature_scaling()

    except Exception as e:
        error_type,error_msg,error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')




