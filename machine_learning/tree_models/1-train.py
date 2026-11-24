#!/usr/bin/env python3
"""Module that trains a tree-based classifier."""


def train_tree(clf, X, y):
    """Fit a tree-based classifier on the given data.

    Args:
        clf: a sklearn classifier instance.
        X: input features.
        y: target labels.
    """
    clf.fit(X, y)
