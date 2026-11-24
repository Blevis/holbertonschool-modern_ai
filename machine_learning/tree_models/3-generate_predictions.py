#!/usr/bin/env python3
"""Module that generates predictions from a trained classifier."""


def generate_predictions(clf, X):
    """Predict class labels for the input samples.

    Args:
        clf: a trained sklearn classifier instance.
        X: feature matrix (numpy.ndarray or pandas.DataFrame).

    Returns:
        A numpy.ndarray of predicted class labels.
    """
    return clf.predict(X)
