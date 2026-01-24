import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('outlier_handling')
from sklearn.ensemble import IsolationForest

class OUTLIER_HANDLING:
    # Directory to save plots
    plot_dir = "plot_outliers"
    os.makedirs(plot_dir, exist_ok=True)

    @staticmethod
    def iqr_capping(X_train, X_test):
        try:
            logger.info('IQR Capping Technique')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            Q1 = X_tr.quantile(0.25)
            Q3 = X_tr.quantile(0.75)

            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            X_tr = X_tr.clip(lower=lower, upper=upper, axis=1)
            X_te = X_te.clip(lower=lower, upper=upper, axis=1)

            logger.info('IQR Capping Technique Completed')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def zscore_capping(X_train, X_test, threshold=3):
        try:
            logger.info('Zscore Capping Technique')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            mean = X_tr.mean()
            std = X_tr.std()

            lower = mean - threshold * std
            upper = mean + threshold * std

            X_tr = X_tr.clip(lower=lower, upper=upper, axis=1)
            X_te = X_te.clip(lower=lower, upper=upper, axis=1)

            logger.info('Zscore Capping Technique Completed')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def mad_capping(X_train, X_test, threshold=3.5):
        try:
            logger.info('MAD Capping Technique')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            median = X_tr.median()
            mad = np.median(np.abs(X_tr - median))

            lower = median - threshold * mad
            upper = median + threshold * mad

            X_tr = X_tr.clip(lower=lower, upper=upper, axis=1)
            X_te = X_te.clip(lower=lower, upper=upper, axis=1)

            logger.info('MAD Capping Technique Completed')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def percentile_capping(X_train, X_test, low=0.01, high=0.99):
        try:
            logger.info('Percentile Capping Technique')

            lower = X_train.quantile(low)
            upper = X_train.quantile(high)

            X_tr = X_train.clip(lower=lower, upper=upper, axis=1)
            X_te = X_test.clip(lower=lower, upper=upper, axis=1)

            logger.info('Percentile Capping Technique Completed')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def winsorization(X_train, X_test):
        try:
            logger.info('Winsorizing Technique')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            numeric_cols = X_tr.select_dtypes(include='number').columns

            lower = X_tr[numeric_cols].quantile(0.05)
            upper = X_tr[numeric_cols].quantile(0.95)

            X_tr[numeric_cols] = X_tr[numeric_cols].clip(lower=lower, upper=upper, axis=1)
            X_te[numeric_cols] = X_te[numeric_cols].clip(lower=lower, upper=upper, axis=1)

            logger.info('Winsorizing Technique Completed')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def clipping(X_train, X_test):
        try:
            logger.info('Clipping Technique')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            numeric_cols = X_tr.select_dtypes(include='number').columns

            X_tr[numeric_cols] = X_tr[numeric_cols].clip(-3, 3)
            X_te[numeric_cols] = X_te[numeric_cols].clip(-3, 3)

            logger.info('Clipping Technique Completed')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def save_outlier_plot(original_train, transformed_train, col_name, technique_name):
        try:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.boxplot(x=original_train[col_name])
            plt.title(f"Before - {col_name}")
            plt.subplot(1, 2, 2)
            sns.boxplot(x=transformed_train[col_name])
            plt.title(f"After - {col_name} ({technique_name})")
            filename = f"{OUTLIER_HANDLING.plot_dir}/{col_name}_{technique_name}.png"
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            logger.info(f"Saved outlier plot: {filename}")

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def save_distribution_plot(original_train, transformed_train, col_name, technique_name):
        try:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.histplot(original_train[col_name], kde=True)
            plt.title(f"Before - {col_name}")
            plt.subplot(1, 2, 2)
            sns.histplot(transformed_train[col_name], kde=True)
            plt.title(f"After - {col_name} ({technique_name})")
            filename = f"{OUTLIER_HANDLING.plot_dir}/{col_name}_{technique_name}_dist.png"
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


    @staticmethod
    def apply_all_techniques(X_train, X_test):
        try:
            techniques = {
                'iqr': OUTLIER_HANDLING.iqr_capping,
                'zscore': OUTLIER_HANDLING.zscore_capping,
                'mad': OUTLIER_HANDLING.mad_capping,
                'percentile': OUTLIER_HANDLING.percentile_capping,
                'winsor': OUTLIER_HANDLING.winsorization,
                'clip': OUTLIER_HANDLING.clipping
            }

            results = {}
            for col in X_train.columns:
                results[col] = {}
                for name, func in techniques.items():
                    X_tr_col, X_te_col = func(X_train[[col]].copy(), X_test[[col]].copy())
                    OUTLIER_HANDLING.save_outlier_plot(X_train[[col]], X_tr_col, col, name)
                    OUTLIER_HANDLING.save_distribution_plot(X_train[[col]], X_tr_col, col, name)
                    results[col][name] = (X_tr_col, X_te_col)

            return results

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

