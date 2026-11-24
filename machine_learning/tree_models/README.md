# Tree-Based Models

Decision trees, pruning, random forests and boosting with Scikit-learn,
XGBoost and LightGBM, trained on the Wine dataset.

| File | Description |
| --- | --- |
| `0-build.py` | `build_decision_tree(min_samples_leaf, min_samples_split, random_state)`: Gini `DecisionTreeClassifier` |
| `1-train.py` | `train_tree(clf, X, y)`: fits a tree-based classifier |
| `2-draw.py` | `draw(clf, feature_names, class_names)`: prints the decision rules |
| `3-generate_predictions.py` | `generate_predictions(clf, X)`: predicted class labels |
| `4-evaluate.py` | `evaluate(true_labels, predicted_labels, class_names)`: classification report |
| `5-pre_pruning.py` | `prepruning(X, y, clf)`: grid search for the best pre-pruning hyperparameters |
| `6-pruning_path.py` | `get_pruning_path(clf, X, y)`: cost-complexity pruning path |
| `7-prune_decision_tree.py` | `prune_and_evaluate_trees(...)`: trains and scores one tree per `ccp_alpha` |
| `8-best_ccp_alpha.py` | `get_best_alpha(clfs, train_scores, test_scores, ccp_alphas)`: best pruning alpha |
| `9-random_forest.py` | `random_forest(n_estimators, random_state)`: `RandomForestClassifier` |
| `10-feature_importance.py` | `feature_importance(rf)`: importances and ascending sort indices |
| `11-boosting.py` | `compare_boosting_classifiers(name, n_estimators, random_state)`: AdaBoost, GradientBoosting, XGBoost or LightGBM |

## Requirements

- Ubuntu 20.04 LTS, Python 3.11, `pycodestyle` 2.14.0
- numpy 2.0.2, scikit-learn 1.6.1, lightgbm 4.5.0, xgboost 2.1.4,
  matplotlib 3.10.0, pillow 11.3.0
