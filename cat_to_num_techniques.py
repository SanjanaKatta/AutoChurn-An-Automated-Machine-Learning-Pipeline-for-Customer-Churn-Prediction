import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('cat_to_num_techniques')

from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from category_encoders import BinaryEncoder,OrdinalEncoder,OneHotEncoder,TargetEncoder,HashingEncoder

class CAT_TO_NUM_TECHNIQUES:
    @staticmethod
    def label_encoding(X_train, X_test):
        try:
            logger.debug('Label Encoding')
            logger.info(f'Before X_train: {X_train.head()}')
            logger.info(f'Before X_test: {X_test.head()}')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            for col in X_tr.columns:
                mapping = {k: i for i, k in enumerate(X_tr[col].unique())}
                X_tr[col] = X_tr[col].map(mapping)
                X_te[col] = X_te[col].map(mapping).fillna(-1)

            logger.debug('After Label Encoding')
            logger.info(f'After X_train: {X_tr.head()}')
            logger.info(f'After X_test: {X_te.head()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def one_hot_encoding(X_train, X_test):
        try:
            logger.debug('One Hot Encoding')
            logger.info(f'Before X_train: {X_train.head()}')
            logger.info(f'Before X_test: {X_test.head()}')

            n_train = X_train.shape[0]
            n_test = X_test.shape[0]

            # Combine train & test to keep same categories
            combined = pd.concat([X_train, X_test], axis=0)

            # One-hot encode
            dummies = pd.get_dummies(combined,drop_first=True)

            # Split back
            X_tr = dummies.iloc[:n_train].reset_index(drop=True)
            X_te = dummies.iloc[n_train:n_train + n_test].reset_index(drop=True)

            logger.debug('After OneHot Encoding')
            logger.info(f'After X_train: {X_tr.head()}')
            logger.info(f'After X_test: {X_te.head()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def ordinal_encoding(X_train, X_test):
        try:
            logger.debug('Ordinal Encoding')
            logger.info(f'Before X_train: {X_train.head()}')
            logger.info(f'Before X_test: {X_test.head()}')

            encoder = OrdinalEncoder(cols=X_train.columns, return_df=True)
            X_tr = encoder.fit_transform(X_train)
            X_te = encoder.transform(X_test)

            logger.debug('After Ordinal Encoding')
            logger.info(f'After X_train: {X_tr.head()}')
            logger.info(f'After X_test: {X_te.head()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def frequency_encoding(X_train, X_test):
        try:
            logger.debug('Frequency Encoding')
            logger.info(f'Before X_train: {X_train.head()}')
            logger.info(f'Before X_test: {X_test.head()}')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            for col in X_tr.columns:
                freq = X_tr[col].value_counts(normalize=True)
                X_tr[col] = X_tr[col].map(freq)
                X_te[col] = X_te[col].map(freq).fillna(0)

            logger.debug('After Frequency Encoding')
            logger.info(f'After X_train: {X_tr.head()}')
            logger.info(f'After X_test: {X_te.head()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def count_encoding(X_train, X_test):
        try:
            logger.debug('Count Encoding')
            logger.info(f'Before X_train: {X_train.head()}')
            logger.info(f'Before X_test: {X_test.head()}')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            for col in X_tr.columns:
                counts = X_tr[col].value_counts()
                X_tr[col] = X_tr[col].map(counts)
                X_te[col] = X_te[col].map(counts).fillna(0)

            logger.debug('After Count Encoding')
            logger.info(f'After X_train: {X_tr.head()}')
            logger.info(f'After X_test: {X_te.head()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def binary_encoding(X_train, X_test):
        try:
            logger.debug('Binary Encoding')
            logger.info(f'Before X_train: {X_train.head()}')
            logger.info(f'Before X_test: {X_test.head()}')

            encoder = BinaryEncoder(cols=X_train.columns, return_df=True)
            X_tr = encoder.fit_transform(X_train)
            X_te = encoder.transform(X_test)

            logger.debug('After Binary Encoding')
            logger.info(f'After X_train: {X_tr.head()}')
            logger.info(f'After X_test: {X_te.head()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def hashing_encoding(X_train, X_test, n_components=8):
        try:
            logger.debug('Hashing Encoding')
            logger.info(f'Before X_train: {X_train.head()}')
            logger.info(f'Before X_test: {X_test.head()}')

            encoder = HashingEncoder(cols=X_train.columns,n_components=n_components,return_df=True)

            X_tr = encoder.fit_transform(X_train)
            X_te = encoder.transform(X_test)

            logger.debug('After Hashing Encoding')
            logger.info(f'After X_train: {X_tr.head()}')
            logger.info(f'After X_test: {X_te.head()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')