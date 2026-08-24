"""
Evaluation metrics for both tasks.

Power prediction (regression): R2, RMSE, MAE.
Fault detection (classification): accuracy, precision, recall, F1, ROC-AUC.
"""
import numpy as np
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score,
)


def regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred, y_proba=None):
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        # MCC and Cohen's kappa are robust to class imbalance -- they expose
        # a "predict-everything-faulty" collapse that F1/accuracy can hide.
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    try:
        if y_proba is not None and len(np.unique(y_true)) > 1:
            out["AUC"] = float(roc_auc_score(y_true, y_proba))
        else:
            out["AUC"] = float("nan")
    except Exception:
        out["AUC"] = float("nan")
    return out
