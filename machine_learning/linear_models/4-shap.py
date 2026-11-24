#!/usr/bin/env python3
"""Module that generates SHAP explanations for a trained model."""
import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """Build a SHAP explainer and compute SHAP values for X_test.

    Args:
        model: a trained regression model.
        X_train: data used as the background dataset of the explainer.
        X_test: data to explain.

    Returns:
        A tuple (explainer, shap_values).
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    return explainer, shap_values
