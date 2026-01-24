import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import sklearn
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from log_code import setup_logging
logger = setup_logging('feature_selection')

from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
reg_constant = VarianceThreshold(threshold=0.0)
reg_quasi_constant = VarianceThreshold(threshold=0.1)
from sklearn.feature_selection import SelectKBest, chi2


class FEATURE_SELECTION():
    @staticmethod
    def feature_selection(X_train, X_test, y_train):
        try:
            logger.info(f"{X_train.columns} -> {X_train.shape}")
            logger.info(f"{X_test.columns} -> {X_test.shape}")

            # ---------------- CONSTANT FEATURES ----------------
            reg_constant.fit(X_train)
            logger.info(f'Columns removed (constant): {X_train.columns[~reg_constant.get_support()]}')

            X_train_fs = pd.DataFrame(reg_constant.transform(X_train),columns=X_train.columns[reg_constant.get_support()])
            X_test_fs = pd.DataFrame(reg_constant.transform(X_test),columns=X_test.columns[reg_constant.get_support()])

            logger.info(f'After Constant Technique: {X_train_fs.columns}  --> {X_train_fs.shape}')
            logger.info(f'After Constant Technique: {X_test_fs.columns}  --> {X_test_fs.shape}')

            # logger.info(X_train_numeric_fs.head(5))
            # logger.info(X_test_numeric_fs.head(5))

            # ---------------- QUASI-CONSTANT FEATURES ----------------
            reg_quasi_constant.fit(X_train_fs)
            logger.info(f'Columns removed (quasi-constant): {X_train_fs.columns[~reg_quasi_constant.get_support()]}')

            X_train_fs_1 = pd.DataFrame(reg_quasi_constant.transform(X_train_fs),columns=X_train_fs.columns[reg_quasi_constant.get_support()])
            X_test_fs_2 = pd.DataFrame(reg_quasi_constant.transform(X_test_fs),columns=X_test_fs.columns[reg_quasi_constant.get_support()])

            logger.info(f'After Quasi Constant Technique: {X_train_fs_1.columns}  --> {X_train_fs_1.shape}')
            logger.info(f'After Quasi Constant Technique: {X_test_fs_2.columns}  --> {X_test_fs_2.shape}')

            # ---------------- CHI-SQUARE TEST ----------------
            X_train_chi = X_train_fs_1.abs()
            X_test_chi = X_test_fs_2.abs()

            skb = SelectKBest(score_func=chi2, k='all')
            skb.fit(X_train_chi, y_train)

            chi_support = skb.get_support()
            chi_features = X_train_fs_1.columns[chi_support]

            X_train_final = X_train_fs_1[chi_features]
            X_test_final = X_test_fs_2[chi_features]

            logger.info(f"After Chi-Square FS : {X_train_final.columns} -> {X_train_final.shape}")

            # ---------------- HYPOTHESIS TESTING (PEARSON) ----------------
            logger.info(f"Before hypothesis testing : {X_train_final.columns} -> {X_train_final.shape}")
            logger.info(f"Before hypothesis testing : {X_test_final.columns} -> {X_test_final.shape}")

            alpha = 0.05
            p_values = []

            # Ensure y is numeric
            if y_train.dtype == "object":
                y_train_numeric = pd.factorize(y_train)[0]
            else:
                y_train_numeric = y_train.values

            for col in X_train_final.columns:
                _, p_value = pearsonr(X_train_final[col].values, y_train_numeric)
                p_values.append(p_value)
                logger.info(f"{col} | p-value = {p_value:.6f}")

            p_values = pd.Series(p_values, index=X_train_final.columns)
            features_to_remove = p_values[p_values > alpha].index.tolist()

            logger.info(f"Features removed by hypothesis testing: {features_to_remove}")

            X_train_final = X_train_final.drop(columns=features_to_remove)
            X_test_final = X_test_final.drop(columns=features_to_remove)

            logger.info(f"After hypothesis testing : {X_train_final.columns} -> {X_train_final.shape}")
            logger.info(f"After hypothesis testing : {X_test_final.columns} -> {X_test_final.shape}")

            return X_train_final, X_test_final

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
