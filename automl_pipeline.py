import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb

def run_pipeline(df, target, problem_type='regression'):
    """
    Strategic ML Engine: Optimized for large datasets and categorical compatibility.
    """
    # 1. Prepare Features and Target
    X = df.drop(columns=[target])
    y = df[target]

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Define Model Library
    # We add enable_categorical=True to XGBoost to match our PreprocessingAgent
    if problem_type == 'classification':
        print(f"🚀 Initializing Classification Pipeline for '{target}'...")
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, solver='lbfgs'),
            "RandomForest": RandomForestClassifier(n_estimators=50, n_jobs=-1),
            "XGBoost": XGBClassifier(
                tree_method="hist", 
                enable_categorical=True, 
                eval_metric='mlogloss'
            ),
            "SVC": SVC(probability=True),
            "LightGBM": lgb.LGBMClassifier(verbosity=-1, importance_type='gain')
        }
    else:
        print(f"🚀 Initializing Regression Pipeline for '{target}'...")
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=50, n_jobs=-1),
            "XGBoost": XGBRegressor(
                tree_method="hist", 
                enable_categorical=True
            ),
            "SVR": SVR(),
            "LightGBM": lgb.LGBMRegressor(verbosity=-1, importance_type='gain')
        }

    results = {}
    best_score = -np.inf
    best_model_name = "None"
    feature_importance = {}

    # 4. Training Loop
    for name, model in models.items():
        try:
            # Performance Guard for SVM
            if name in ["SVC", "SVR"] and len(df) > 20000:
                print(f"⏩ Skipping {name}: Dataset too large (>20k rows).")
                continue

            # Fit Model
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            results[name] = score
            print(f"✅ {name} Score: {score:.4f}")

            # Track the winner and extract importance
            if score > best_score:
                best_score = score
                best_model_name = name
                
                # --- ROBUST FEATURE IMPORTANCE EXTRACTION ---
                if hasattr(model, 'feature_importances_'):
                    # Tree-based (RF, XGB, LGBM)
                    importances = model.feature_importances_
                    feature_importance = dict(zip(X.columns, importances.astype(float)))
                
                elif hasattr(model, 'coef_'):
                    # Linear-based (Linear/Logistic)
                    coeffs = model.coef_
                    # Logistic coef_ is often 2D (for multiclass), flatten if necessary
                    if len(coeffs.shape) > 1:
                        coeffs = np.mean(np.abs(coeffs), axis=0)
                    else:
                        coeffs = np.abs(coeffs)
                    feature_importance = dict(zip(X.columns, coeffs.astype(float)))

        except Exception as e:
            print(f"⚠️ Error training {name}: {str(e)}")

    # 5. Finalize Results
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

    return {
        "best_model": best_model_name,
        "score": float(best_score),  # Changed 'best_score' to 'score'
        "metrics": {k: float(v) for k, v in results.items()}, # Changed 'all_scores' to 'metrics'
        "feature_importance": sorted_importance, # Changed 'feature_importance_dict' to 'feature_importance'
        "target": target, # Added target so analyst_agent knows what it's looking at
        "problem_type": problem_type
    }