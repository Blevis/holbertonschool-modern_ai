#!/usr/bin/env python3
"""Module that builds a decision tree classifier."""
from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """Create an untrained Gini decision tree classifier with no max depth.

    Args:
        min_samples_leaf: minimum number of samples required at a leaf node.
        min_samples_split: minimum number of samples required to split an
            internal node.
        random_state: seed used by the random number generator.

    Returns:
        A sklearn DecisionTreeClassifier instance.
    """
    return tree.DecisionTreeClassifier(criterion="gini",
                                       max_depth=None,
                                       min_samples_leaf=min_samples_leaf,
                                       min_samples_split=min_samples_split,
                                       random_state=random_state)
