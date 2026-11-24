#!/usr/bin/env python3
"""Module that evaluates a classifier with a classification report."""
from sklearn import metrics


def evaluate(true_labels, predicted_labels, class_names):
    """Build a classification report with precision, recall and F1-score.

    Args:
        true_labels: ground truth labels.
        predicted_labels: predicted labels.
        class_names: list of class names matching the label indices.

    Returns:
        The classification report as a string.
    """
    return metrics.classification_report(true_labels, predicted_labels,
                                         target_names=class_names)
