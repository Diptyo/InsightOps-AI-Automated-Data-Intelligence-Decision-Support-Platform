"""
Provide ML models and hyperparameter grids.
"""

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

def get_models(problem):
    if problem == "regression":
        return {
            "LinearRegression": (
                LinearRegression(),
                {} # Typically no hyperparameters to tune for basic LR
            ),
            "DecisionTree": (
                DecisionTreeRegressor(max_depth=5, random_state=42),
                {"max_depth": [3, 5, 10], "min_samples_split": [2, 10]}
            ),
            "RandomForest": (
                RandomForestRegressor(),
                {"n_estimators": [100, 200], "max_depth": [None, 10]}
            ),
            "GradientBoost": (
                GradientBoostingRegressor(),
                {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]}
            ),
            "XGBoost": (
                XGBRegressor(random_state=42),
                {"n_estimators": [100, 200], "max_depth": [3, 6]}
            ),
            "LightGBM": (
                LGBMRegressor(random_state=42),
                {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]}
            ),
            "SVR": (
                SVR(),
                {"C": [0.1, 1, 10]}
            )
        }
    else: # Classification
        return {
            "LogisticRegression": (
                LogisticRegression(max_iter=1000),
                {"C": [0.1, 1, 10]}
            ),
            "DecisionTree": (
                DecisionTreeClassifier(max_depth=5, random_state=42),
                {"max_depth": [3, 5, 10], "criterion": ["gini", "entropy"]}
            ),
            "RandomForest": (
                RandomForestClassifier(),
                {"n_estimators": [100, 200]}
            ),
            "GradientBoost": (
                GradientBoostingClassifier(),
                {"n_estimators": [100, 200]}
            ),
            "XGBoost": (
                XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                {"n_estimators": [100, 200]}
            ),
            "LightGBM": (
                LGBMClassifier(random_state=42),
                {"n_estimators": [100, 200]}
            ),
            "SVC": (
                SVC(probability=True),
                {"C": [0.1, 1, 10]}
            )
        }