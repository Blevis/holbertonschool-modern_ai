#!/usr/bin/env python3
"""Module that computes feature importances of a random forest."""
import numpy as np


def feature_importance(rf):
    """Get feature importances and their ascending sort order.

    Args:
        rf: a trained sklearn RandomForestClassifier instance.

    Returns:
        A tuple (importances, indices) where indices sorts the features
        from least to most important.
    """
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    return importances, indices
