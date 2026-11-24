#!/usr/bin/env python3
"""Module that builds Support Vector Machine classifiers."""
from sklearn import svm


def get_SVM_model(name, random_state):
    """Create an untrained SVM classifier with the requested kernel.

    Args:
        name: kernel type, one of 'linear', 'poly' or 'rbf'.
        random_state: seed used by the random number generator.

    Returns:
        An untrained sklearn SVC instance.
    """
    return svm.SVC(kernel=name, random_state=random_state)
