"""
Train multiple ML models efficiently.
Updated for memory-safety and dataset-agnostic performance.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
# SWAPPED: Legacy GradientBoosting for modern Histogram-based versions
from sklearn.ensemble import (
    RandomForestRegressor, 
    RandomForestClassifier, 
    HistGradientBoostingRegressor, 
    HistGradientBoostingClassifier
)

# Optional heavy-duty models
try: 
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except ImportError: 
    XGB_AVAILABLE = False

try: 
    from lightgbm import LGBMRegressor, LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError: 
    LGBM_AVAILABLE = False


def train_models(X, y, problem_type):
    """
    Train multiple models and return their scores.
    Uses memory-efficient algorithms for large generic datasets.
    """

    # 1. Generic Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # =========================
    # REGRESSION MODELS
    # =========================
    if problem_type == "regression":
        models = {
            "LinearRegression": LinearRegression(),
            # HistGradientBoosting handles millions of rows using minimal RAM
            "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=100, early_stopping=True),
            # Limited RandomForest to prevent memory spikes on generic data
            "RandomForest": RandomForestRegressor(n_estimators=30, max_depth=10, n_jobs=-1),
        }

        if XGB_AVAILABLE: 
            # 'hist' method makes XGBoost act like LightGBM (very fast/light)
            models["XGBoost"] = XGBRegressor(n_jobs=-1, tree_method='hist')

        if LGBM_AVAILABLE: 
            models["LightGBM"] = LGBMRegressor(verbose=-1, n_jobs=-1)

    # =========================
    # CLASSIFICATION MODELS
    # =========================
    else:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1),
            "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, early_stopping=True),
            "RandomForest": RandomForestClassifier(n_estimators=30, max_depth=10, n_jobs=-1),
        }

        if XGB_AVAILABLE:
            models["XGBoost"] = XGBClassifier(n_jobs=-1, eval_metric='mlogloss', tree_method='hist')

        if LGBM_AVAILABLE:
            models["LightGBM"] = LGBMClassifier(n_jobs=-1, verbose=-1)

    # 2. Universal Training Loop with Error Handling
    for name, model in models.items():
        try:
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            if problem_type == "regression":
                score = r2_score(y_test, preds)
            else:
                score = accuracy_score(y_test, preds)
            
            # 🔥 KEY CHANGE: Store both score AND model object
            results[name] = (score, model) 
            
            print(f"✅ {name} Score: {score:.4f}")

        except Exception as e:
            print(f"⚠️ Could not train {name}: {e}")

    return results