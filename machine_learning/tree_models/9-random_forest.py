#!/usr/bin/env python3
"""Module that builds a random forest classifier."""
from sklearn import ensemble


def random_forest(n_estimators, random_state):
    """Create an untrained random forest classifier.

    Args:
        n_estimators: number of trees in the forest.
        random_state: seed used by the random number generator.

    Returns:
        A sklearn RandomForestClassifier instance.
    """
    return ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                           random_state=random_state)
