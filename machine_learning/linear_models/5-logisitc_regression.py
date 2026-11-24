#!/usr/bin/env python3
"""Module that builds a logistic regression classifier."""
from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """Create an untrained logistic regression classifier.

    Args:
        random_state: integer seed used for reproducibility.

    Returns:
        An untrained sklearn LogisticRegression instance.
    """
    return linear_model.LogisticRegression(random_state=random_state)
