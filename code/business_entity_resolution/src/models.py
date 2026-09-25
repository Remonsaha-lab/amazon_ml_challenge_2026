"""
Machine Learning Matcher Models for Amazon ML Challenge 2026.
Supports head-to-head benchmarking between LightGBM and XGBoost
on identical entity-grouped splits.
"""

from typing import Dict, Any, Tuple
import numpy as np
import lightgbm as lgb
import xgboost as xgb

from features import FEATURE_NAMES


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    params: Dict[str, Any] = None
) -> lgb.LGBMClassifier:
    """Train LightGBM binary classifier on pairwise features."""
    default_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    if params:
        default_params.update(params)
        
    model = lgb.LGBMClassifier(**default_params)
    
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
    else:
        model.fit(X_train, y_train)
        
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    params: Dict[str, Any] = None
) -> xgb.XGBClassifier:
    """Train XGBoost binary classifier on pairwise features."""
    default_params = {
        "objective": "binary:logistic",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1
    }
    if params:
        default_params.update(params)
        
    model = xgb.XGBClassifier(**default_params)
    
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
        
    return model


def get_feature_importances(model, feature_names=FEATURE_NAMES) -> Dict[str, float]:
    """Extract and sort feature importances from a trained model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # Normalize to percentage
        total = sum(importances)
        if total > 0:
            norm_imp = [x / total for x in importances]
        else:
            norm_imp = importances
        return dict(sorted(zip(feature_names, norm_imp), key=lambda x: x[1], reverse=True))
    return {}
