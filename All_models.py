import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pickle

from log_code import setup_logging
logger = setup_logging('All_Models')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import GridSearchCV

def knn(X_train,y_train,X_test,y_test):
    try:
      global knn_reg
      knn_reg = KNeighborsClassifier(n_neighbors=5)
      knn_reg.fit(X_train,y_train)
      logger.info(f'KNN Test Accuracy : {accuracy_score(y_test,knn_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

def nb(X_train,y_train,X_test,y_test):
    try:
      global naive_reg
      naive_reg = GaussianNB()
      naive_reg.fit(X_train,y_train)
      logger.info(f'Naive Bayes Test Accuracy : {accuracy_score(y_test,naive_reg.predict(X_test))}')

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')


def lr(X_train, y_train, X_test, y_test):
    try:
        global lr_reg
        lr_reg = LogisticRegression(max_iter=1000)
        lr_reg.fit(X_train, y_train)

        y_pred = lr_reg.predict(X_test)
        y_prob = lr_reg.predict_proba(X_test)[:, 1]

        logger.info(f'LogisticRegression Test Accuracy : {accuracy_score(y_test, y_pred)}')
        #logger.info(f'LogisticRegression ROC-AUC : {roc_auc_score(y_test, y_prob)}')

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.error(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

def dt(X_train,y_train,X_test,y_test):
    try:
      global dt_reg
      dt_reg = DecisionTreeClassifier(criterion='entropy')
      dt_reg.fit(X_train,y_train)
      logger.info(f'DecisionTreeClassifier Test Accuracy : {accuracy_score(y_test,dt_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

def rf(X_train,y_train,X_test,y_test):
    try:
      global rf_reg
      rf_reg = RandomForestClassifier(n_estimators=5,criterion='entropy')
      rf_reg.fit(X_train,y_train)
      logger.info(f'RandomForestClassifier Test Accuracy : {accuracy_score(y_test,rf_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

def ada(X_train,y_train,X_test,y_test):
    try:
      global ada_reg
      t = LogisticRegression()
      ada_reg = AdaBoostClassifier(estimator=t,n_estimators=5)
      ada_reg.fit(X_train,y_train)
      logger.info(f'AdaBoostClassifier Test Accuracy : {accuracy_score(y_test,ada_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

def gb(X_train,y_train,X_test,y_test):
    try:
      global gb_reg
      gb_reg = GradientBoostingClassifier(n_estimators=5)
      gb_reg.fit(X_train,y_train)
      logger.info(f'GradientBoostingClassifier Test Accuracy : {accuracy_score(y_test,gb_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

def xgb_(X_train,y_train,X_test,y_test):
    try:
      global xg_reg
      xg_reg = XGBClassifier()
      xg_reg.fit(X_train,y_train)
      logger.info(f'XGBClassifier Test Accuracy : {accuracy_score(y_test,xg_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

def svm_c(X_train,y_train,X_test,y_test):
    try:
      global svm_reg
      svm_reg = SVC(kernel='rbf',probability=True)
      svm_reg.fit(X_train,y_train)
      logger.info(f'SVM Test Accuracy : {accuracy_score(y_test,svm_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

def common(X_train,y_train,X_test,y_test):
    try:
      logger.info('=========KNeighborsClassifier===========')
      knn(X_train,y_train,X_test,y_test)
      logger.info('=========GaussianNB===========')
      nb(X_train,y_train,X_test,y_test)
      logger.info('=========LogisticRegression===========')
      lr(X_train,y_train,X_test,y_test)
      logger.info('=========DecisionTreeClassifier===========')
      dt(X_train,y_train,X_test,y_test)
      logger.info('=========RandomForestClassifier===========')
      rf(X_train,y_train,X_test,y_test)
      logger.info('=========AdabostClassifier===========')
      ada(X_train,y_train,X_test,y_test)
      logger.info('=========GradientBoostingClassifier===========')
      gb(X_train,y_train,X_test,y_test)
      logger.info('=========XGBClassifier===========')
      xgb_(X_train,y_train,X_test,y_test)
      logger.info('=========SVM===========')
      svm_c(X_train,y_train,X_test,y_test)

      #Predictions for each algorithm
      knn_predictions = knn_reg.predict_proba(X_test)[:, 1]
      naive_predictions = naive_reg.predict_proba(X_test)[:, 1]
      lr_predictions = lr_reg.predict_proba(X_test)[:, 1]
      dt_predictions = dt_reg.predict_proba(X_test)[:, 1]
      rf_predictions = rf_reg.predict_proba(X_test)[:, 1]
      ada_predictions = ada_reg.predict_proba(X_test)[:, 1]
      gb_predictions = gb_reg.predict_proba(X_test)[:, 1]
      xgb_predictions = xg_reg.predict_proba(X_test)[:, 1]
      svm_predictions = svm_reg.predict(X_test)

      #ROC Curve
      knn_fpr, knn_tpr, knn_thre = roc_curve(y_test, knn_predictions)
      nb_fpr, nb_tpr, nb_thre = roc_curve(y_test, naive_predictions)
      lr_fpr, lr_tpr, lr_thre = roc_curve(y_test, lr_predictions)
      dt_fpr, dt_tpr, dt_thre = roc_curve(y_test, dt_predictions)
      rf_fpr, rf_tpr, rf_thre = roc_curve(y_test, rf_predictions)
      ada_fpr, ada_tpr, ada_thre = roc_curve(y_test, ada_predictions)
      gb_fpr, gb_tpr, gb_thre = roc_curve(y_test, gb_predictions)
      xgb_fpr, xgb_tpr, xgb_thre = roc_curve(y_test, xgb_predictions)
      svm_fpr, svm_tpr, svm_thre = roc_curve(y_test, svm_predictions)

      plt.figure(figsize=(5, 3))
      plt.plot([0, 1], [0, 1], "k--")

      model_outputs = {
          "KNN": (knn_reg.predict_proba(X_test)[:, 1], knn_reg.predict(X_test)),
          "Naive Bayes": (naive_reg.predict_proba(X_test)[:, 1], naive_reg.predict(X_test)),
          "Logistic Regression": (lr_reg.predict_proba(X_test)[:, 1], lr_reg.predict(X_test)),
          "Decision Tree": (dt_reg.predict_proba(X_test)[:, 1], dt_reg.predict(X_test)),
          "Random Forest": (rf_reg.predict_proba(X_test)[:, 1], rf_reg.predict(X_test)),
          "AdaBoost": (ada_reg.predict_proba(X_test)[:, 1], ada_reg.predict(X_test)),
          "Gradient Boosting": (gb_reg.predict_proba(X_test)[:, 1], gb_reg.predict(X_test)),
          "XGBoost": (xg_reg.predict_proba(X_test)[:, 1], xg_reg.predict(X_test)),
          "SVM": (svm_reg.predict_proba(X_test)[:, 1], svm_reg.predict(X_test))
      }

      auc_scores = {}
      acc_scores = {}

      for name, (prob_pred, class_pred) in model_outputs.items():
          fpr, tpr, _ = roc_curve(y_test, prob_pred)

          auc = roc_auc_score(y_test, prob_pred)
          acc = accuracy_score(y_test, class_pred)

          auc_scores[name] = auc
          acc_scores[name] = acc

          plt.plot(fpr, tpr, label=f'{name} (Acc = {acc:.4f}')

      plt.xlabel("FPR")
      plt.ylabel("TPR")
      plt.title("ROC Curve - All Models")
      plt.legend(loc=0)
      plt.show()

      # ------------------ ROC-AUC SCORE CALCULATION ------------------
      for model, score in auc_scores.items():
          logger.info(f"{model} ROC-AUC Score: {score}")

          # ------------------ SELECT BEST MODEL ------------------
      best_model_name = max(auc_scores, key=auc_scores.get)
      best_auc = auc_scores[best_model_name]
      logger.info("===================================")
      logger.info(f"BEST MODEL: {best_model_name}")
      logger.info(f"BEST ROC-AUC: {best_auc}")
      logger.info("===================================")

      # Map model name to object
      model_dict = {
          "KNN": knn_reg,
          "Naive Bayes": naive_reg,
          "Logistic Regression": lr_reg,
          "Decision Tree": dt_reg,
          "Random Forest": rf_reg,
          "AdaBoost": ada_reg,
          "Gradient Boosting": gb_reg,
          "XGBoost": xg_reg,
          "SVM": svm_reg
      }

      best_model = model_dict[best_model_name]

      logger.info(f"Tuning Logistic | X_train: {X_train.shape}, y_train: {y_train.shape}")
      logger.info(f"Tuning Logistic | X_test: {X_test.shape}, y_test: {y_test.shape}")

      if best_model_name == "Naive Bayes":
          logger.info("Tuning BEST model: Naive Bayes")
          best_model = tune_naive_bayes(X_train, y_train, X_test, y_test)
      else:
          best_model = model_dict[best_model_name]

      # ------------------ SAVE BEST MODEL ------------------
      with open("Churn_Prediction_Best_Model.pkl", "wb") as f:
          pickle.dump(best_model, f)

      logger.info(f"{best_model_name} saved successfully as Churn_Prediction_Best_Model.pkl")

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

def tune_naive_bayes(X_train, y_train, X_test, y_test):
    try:
        param_grid = {
            'var_smoothing': [1e-12, 1e-10, 1e-9, 1e-8, 1e-7]
        }

        grid = GridSearchCV(GaussianNB(),param_grid=param_grid,cv=5,scoring='roc_auc',n_jobs=-1)

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        logger.info(f"Best Parameters: {grid.best_params_}")
        logger.info(f"Best CV ROC-AUC: {grid.best_score_}")

        # -------- TEST SET EVALUATION --------
        y_test_pred = best_model.predict(X_test)
        y_test_prob = best_model.predict_proba(X_test)[:, 1]

        test_acc = accuracy_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_prob)

        logger.info(f"Tuned Naive Bayes Test Accuracy: {test_acc}")
        logger.info(f"Tuned Naive Bayes Test ROC-AUC: {test_auc}")

        return best_model

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')