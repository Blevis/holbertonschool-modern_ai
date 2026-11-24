#!/usr/bin/env python3
"""Module that builds a Lasso (L1 regularized) regression model."""
from sklearn import linear_model


def lasso_regression(random_state):
    """Create an untrained Lasso regression model.

    Args:
        random_state: integer seed used for reproducibility.

    Returns:
        An untrained sklearn Lasso instance.
    """
    return linear_model.Lasso(random_state=random_state)
