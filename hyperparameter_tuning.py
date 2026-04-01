"""
Hyperparameter tuning using GridSearchCV.
"""

from sklearn.model_selection import GridSearchCV


def tune_model(model, params, X, y, problem):

    metric = "accuracy" if problem=="classification" else "r2"

    grid = GridSearchCV(
        model,
        params,
        scoring=metric,
        cv=3,
        n_jobs=-1
    )

    grid.fit(X,y)

    return grid.best_estimator_, grid.best_score_