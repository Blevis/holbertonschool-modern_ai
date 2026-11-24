#!/usr/bin/env python3
"""Module that trains and evaluates post-pruned decision trees."""
from sklearn import tree
train_tree = __import__('1-train').train_tree


def prune_and_evaluate_trees(X_train, y_train, X_test, y_test, ccp_alphas,
                             random_state, min_samples_leaf,
                             min_samples_split):
    """Train one decision tree per ccp_alpha and score each one.

    Args:
        X_train: training features.
        y_train: training labels.
        X_test: testing features.
        y_test: testing labels.
        ccp_alphas: numpy.ndarray of pruning alpha values.
        random_state: integer seed for reproducibility.
        min_samples_leaf: minimum number of samples at a leaf node.
        min_samples_split: minimum number of samples to split a node.

    Returns:
        A tuple (clfs, train_scores, test_scores).
    """
    clfs, train_scores, test_scores = [], [], []
    for alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=random_state,
                                          min_samples_leaf=min_samples_leaf,
                                          min_samples_split=min_samples_split,
                                          ccp_alpha=alpha)
        train_tree(clf, X_train, y_train)
        clfs.append(clf)
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))
    return clfs, train_scores, test_scores
