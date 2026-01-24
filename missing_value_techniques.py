import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from log_code import setup_logging
logger = setup_logging("missing_values")

np.random.seed(42)

class MISSING_VALUES:
    @staticmethod
    def mean_imputation(X_train, X_test):
        logger.info("Mean Imputation")
        num_cols = X_train.select_dtypes(exclude="object").columns

        imp = SimpleImputer(strategy="mean")
        X_train[num_cols] = imp.fit_transform(X_train[num_cols])
        X_test[num_cols] = imp.transform(X_test[num_cols])

        return X_train, X_test

    @staticmethod
    def median_imputation(X_train, X_test):
        logger.info("Median Imputation")
        num_cols = X_train.select_dtypes(exclude="object").columns

        imp = SimpleImputer(strategy="median")
        X_train[num_cols] = imp.fit_transform(X_train[num_cols])
        X_test[num_cols] = imp.transform(X_test[num_cols])

        return X_train, X_test

    @staticmethod
    def mode_imputation(X_train, X_test):
        logger.info("Mode Imputation")

        imp = SimpleImputer(strategy="most_frequent")
        X_train[:] = imp.fit_transform(X_train)
        X_test[:] = imp.transform(X_test)

        return X_train, X_test

    @staticmethod
    def constant_imputation(X_train, X_test, value=-999):
        logger.info("Constant Imputation")

        for col in X_train.columns:
            if pd.api.types.is_numeric_dtype(X_train[col]):
                X_train[col].fillna(value, inplace=True)
                X_test[col].fillna(value, inplace=True)
            else:
                X_train[col].fillna("Unknown", inplace=True)
                X_test[col].fillna("Unknown", inplace=True)

        return X_train, X_test

    @staticmethod
    def random_imputation(X_train, X_test):
        logger.info("Random Sample Imputation")

        for col in X_train.columns:
            train_non_null = X_train[col].dropna()
            if train_non_null.empty:
                continue

            X_train[col] = X_train[col].apply(
                lambda x: np.random.choice(train_non_null) if pd.isna(x) else x
            )
            X_test[col] = X_test[col].apply(
                lambda x: np.random.choice(train_non_null) if pd.isna(x) else x
            )

        return X_train, X_test

    @staticmethod
    def knn_imputation(X_train, X_test):
        logger.info("KNN Imputation")

        num_cols = X_train.select_dtypes(exclude="object").columns
        cat_cols = X_train.select_dtypes(include="object").columns

        knn = KNNImputer(n_neighbors=5)
        X_train[num_cols] = knn.fit_transform(X_train[num_cols])
        X_test[num_cols] = knn.transform(X_test[num_cols])

        for col in cat_cols:
            mode = X_train[col].mode()[0]
            X_train[col].fillna(mode, inplace=True)
            X_test[col].fillna(mode, inplace=True)

        return X_train, X_test

    @staticmethod
    def iterative_imputation(X_train, X_test):
        logger.info("Iterative (MICE) Imputation")

        num_cols = X_train.select_dtypes(exclude="object").columns
        cat_cols = X_train.select_dtypes(include="object").columns

        mice = IterativeImputer(
            random_state=42,
            max_iter=10,
            sample_posterior=True
        )

        X_train[num_cols] = mice.fit_transform(X_train[num_cols])
        X_test[num_cols] = mice.transform(X_test[num_cols])

        for col in cat_cols:
            mode = X_train[col].mode()[0]
            X_train[col].fillna(mode, inplace=True)
            X_test[col].fillna(mode, inplace=True)

        return X_train, X_test

    @staticmethod
    def forward_fill(X_train, X_test):
        try:
            logger.info('Forward Fill Imputation started')
            logger.info(f'Before imputation X_train:\n{X_train.isnull().sum()}')
            logger.info(f'Before imputation X_test:\n{X_test.isnull().sum()}')

            X_tr = X_train.copy().ffill()
            X_te = X_test.copy().ffill()

            logger.info('Forward Fill Imputation completed')
            logger.info(f'After imputation X_train:\n{X_tr.isnull().sum()}')
            logger.info(f'After imputation X_test:\n{X_te.isnull().sum()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(
                f'Error in forward_fill at line {error_line.tb_lineno}: {error_msg}'
            )

    @staticmethod
    def backward_fill(X_train, X_test):
        try:
            logger.info('Backward Fill Imputation started')
            logger.info(f'Before imputation X_train:\n{X_train.isnull().sum()}')
            logger.info(f'Before imputation X_test:\n{X_test.isnull().sum()}')

            X_tr = X_train.copy().bfill()
            X_te = X_test.copy().bfill()

            logger.info('Backward Fill Imputation completed')
            logger.info(f'After imputation X_train:\n{X_tr.isnull().sum()}')
            logger.info(f'After imputation X_test:\n{X_te.isnull().sum()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(
                f'Error in backward_fill at line {error_line.tb_lineno}: {error_msg}'
            )

    @staticmethod
    def interpolation(X_train, X_test, method="linear"):
        try:
            logger.info('Interpolation Imputation started')
            logger.info(f'Before imputation X_train:\n{X_train.isnull().sum()}')
            logger.info(f'Before imputation X_test:\n{X_test.isnull().sum()}')

            X_tr = X_train.copy()
            X_te = X_test.copy()

            num_cols = X_tr.select_dtypes(include=["int64", "float64"]).columns
            cat_cols = X_tr.select_dtypes(include=["object"]).columns

            # Interpolate only numeric columns
            X_tr[num_cols] = X_tr[num_cols].interpolate(
                method=method, limit_direction="both"
            )
            X_te[num_cols] = X_te[num_cols].interpolate(
                method=method, limit_direction="both"
            )

            # Categorical → mode
            for col in cat_cols:
                X_tr[col].fillna(X_tr[col].mode()[0], inplace=True)
                X_te[col].fillna(X_te[col].mode()[0], inplace=True)

            logger.info('Interpolation Imputation completed')
            logger.info(f'After imputation X_train:\n{X_tr.isnull().sum()}')
            logger.info(f'After imputation X_test:\n{X_te.isnull().sum()}')

            return X_tr, X_te

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(
                f'Error in interpolation at line {error_line.tb_lineno}: {error_msg}'
            )

