"""
Model zoo -- the baselines from the anchor paper (Abdelsattar et al., 2025).

CatBoost and XGBoost are optional: if not installed, the code prints a notice
and continues with the scikit-learn models, so the pipeline always runs.
HistGradientBoosting is a native, fast boosting stand-in that is always present.
"""
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

import config

_RS = config.RANDOM_STATE

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False

try:
    from xgboost import XGBRegressor, XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def get_regressors():
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=_RS, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=_RS),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=_RS),
        "LinearRegression": LinearRegression(),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, random_state=_RS, n_jobs=-1,
        )
    if _HAS_CATBOOST:
        models["CatBoost"] = CatBoostRegressor(
            iterations=600, learning_rate=0.05, depth=6, random_state=_RS, verbose=0,
        )
    return models


def get_classifiers():
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=_RS, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=_RS),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=_RS),
        "LogisticRegression": LogisticRegression(max_iter=1000),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, random_state=_RS,
            n_jobs=-1, eval_metric="logloss",
        )
    if _HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=600, learning_rate=0.05, depth=6, random_state=_RS, verbose=0,
        )
    return models


def availability_note():
    missing = []
    if not _HAS_CATBOOST:
        missing.append("catboost")
    if not _HAS_XGB:
        missing.append("xgboost")
    if missing:
        return f"[models] Optional libs not found ({', '.join(missing)}); running sklearn models only. `pip install {' '.join(missing)}` to enable them."
    return "[models] CatBoost + XGBoost available."
