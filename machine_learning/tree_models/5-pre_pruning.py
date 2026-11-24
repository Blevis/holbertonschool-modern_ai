#!/usr/bin/env python3
"""Module that grid searches pre-pruning hyperparameters of a tree."""
from sklearn import model_selection


def prepruning(X, y, clf):
    """Find the best pre-pruning hyperparameters with a grid search.

    Args:
        X: input features.
        y: target labels.
        clf: an untrained sklearn DecisionTreeClassifier instance.

    Returns:
        A dictionary with the best combination of hyperparameters.
    """
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": list(range(2, 5)),
        "min_samples_leaf": list(range(2, 5)),
        "min_samples_split": list(range(2, 5)),
    }
    grid = model_selection.GridSearchCV(clf, param_grid)
    grid.fit(X, y)
    return grid.best_params_
