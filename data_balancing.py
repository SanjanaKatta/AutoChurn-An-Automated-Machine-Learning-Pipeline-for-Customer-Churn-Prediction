import numpy as np
import pandas as pd
import sys
from imblearn.over_sampling import SMOTE
from log_code import setup_logging

logger = setup_logging("data_balancing")


class DATA_BALANCING:

    @staticmethod
    def data_balancing(X_train, y_train, X_test, force_balance=False):
        try:
            logger.info("Checking class imbalance")

            count_0 = sum(y_train == 0)
            count_1 = sum(y_train == 1)

            logger.info(f"Class 0 (No) : {count_0}")
            logger.info(f"Class 1 (Yes): {count_1}")

            # Avoid division error
            if count_1 == 0 or count_0 == 0:
                logger.warning("Only one class present. Skipping balancing.")
                return X_train, y_train, X_test

            imbalance_ratio = max(count_0, count_1) / min(count_0, count_1)
            logger.info(f"Imbalance Ratio: {imbalance_ratio:.2f}")

            # ===============================
            # DECISION LOGIC
            # ===============================
            if imbalance_ratio < 3 and not force_balance:
                logger.info(
                    "Imbalance ratio is acceptable (<3). "
                    "Skipping data balancing to avoid noise."
                )
                return X_train, y_train, X_test

            logger.info("Applying SMOTE for data balancing")

            smote = SMOTE(random_state=42)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

            logger.info("After SMOTE:")
            logger.info(f"Class 0 (No) : {sum(y_train_bal == 0)}")
            logger.info(f"Class 1 (Yes): {sum(y_train_bal == 1)}")

            return X_train_bal, y_train_bal, X_test

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(
                f"Error in data_balancing at line {error_line.tb_lineno}: {error_msg}"
            )
            return X_train, y_train, X_test
