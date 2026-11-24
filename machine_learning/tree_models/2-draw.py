#!/usr/bin/env python3
"""Module that prints the decision rules of a trained decision tree."""
from sklearn import tree


def draw(clf, feature_names, class_names):
    """Print a text representation of a trained decision tree.

    Args:
        clf: a trained sklearn DecisionTreeClassifier instance.
        feature_names: list of the input feature names.
        class_names: list of the target class names.
    """
    print(tree.export_text(clf, feature_names=list(feature_names),
                           class_names=list(class_names)))
