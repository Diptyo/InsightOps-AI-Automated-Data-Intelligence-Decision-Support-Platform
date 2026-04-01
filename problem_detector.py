"""
Detect problem type (classification or regression)
"""

import numpy as np


def detect_problem_type(y):
    """
    Determine ML problem type intelligently
    """

    unique_ratio = y.nunique() / len(y)

    # Classification
    if unique_ratio < 0.1 or y.nunique() <= 15:
        return "classification"

    # Regression
    if np.issubdtype(y.dtype, np.number):
        return "regression"

    # Default fallback
    return "classification"