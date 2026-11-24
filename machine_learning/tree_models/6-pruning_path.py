#!/usr/bin/env python3
"""Module that retrieves the cost-complexity pruning path of a tree."""


def get_pruning_path(clf, X, y):
    """Compute the cost-complexity pruning path of a decision tree.

    Args:
        clf: a sklearn DecisionTreeClassifier instance.
        X: input features.
        y: target labels.

    Returns:
        A tuple (ccp_alphas, impurities) of numpy.ndarray.
    """
    path = clf.cost_complexity_pruning_path(X, y)
    return path.ccp_alphas, path.impurities
