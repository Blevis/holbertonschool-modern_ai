#!/usr/bin/env python3
"""Module that builds a Ridge (L2 regularized) regression model."""
from sklearn import linear_model


def ridge_regression(random_state):
    """Create an untrained Ridge regression model.

    Args:
        random_state: integer seed used for reproducibility.

    Returns:
        An untrained sklearn Ridge instance.
    """
    return linear_model.Ridge(random_state=random_state)
