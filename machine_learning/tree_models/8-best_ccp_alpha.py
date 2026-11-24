#!/usr/bin/env python3
"""Module that selects the best ccp_alpha for a pruned decision tree."""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """Select the best pruning alpha and its classifier.

    The best model has the highest test accuracy; ties are broken by the
    smallest train/test accuracy gap, then by the largest ccp_alpha.

    Args:
        clfs: list of trained DecisionTreeClassifier instances.
        train_scores: list of training accuracies for each classifier.
        test_scores: list of testing accuracies for each classifier.
        ccp_alphas: list or array of ccp_alpha values of the classifiers.

    Returns:
        A tuple (best_alpha, best_clf).
    """
    best = max(range(len(clfs)),
               key=lambda i: (test_scores[i],
                              -abs(train_scores[i] - test_scores[i]),
                              ccp_alphas[i]))
    return ccp_alphas[best], clfs[best]
