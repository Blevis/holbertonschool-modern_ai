#!/usr/bin/env python3
"""Module that builds an ordinary least squares linear regression model."""
from sklearn import linear_model


def Linear_Regression():
    """Create an untrained ordinary least squares linear regression model.

    Returns:
        An untrained sklearn LinearRegression instance.
    """
    return linear_model.LinearRegression()
