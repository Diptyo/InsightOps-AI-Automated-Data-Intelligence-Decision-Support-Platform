"""
Explain model using SHAP.
"""

import shap
import pandas as pd


def explain_model(model,X):

    explainer=shap.Explainer(model,X)

    shap_values=explainer(X)

    importance=abs(shap_values.values).mean(axis=0)

    return pd.Series(
    importance,
    index=X.columns
    ).sort_values(ascending=False)