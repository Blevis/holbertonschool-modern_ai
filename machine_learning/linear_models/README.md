# Linear Models

Linear regression, regularization, explainability and linear classifiers with
Scikit-learn and SHAP.

| File | Description |
| --- | --- |
| `0-linear_regression.py` | `Linear_Regression()`: untrained OLS `LinearRegression` model |
| `1-regression_evaluation_metrics.py` | `evaluation_metrics_for_regression(y_true, y_pred)`: returns `(mse, rmse, mae, r2)` |
| `2-ridge_regression.py` | `ridge_regression(random_state)`: untrained L2-regularized `Ridge` model |
| `3-Lasso_regression.py` | `lasso_regression(random_state)`: untrained L1-regularized `Lasso` model |
| `4-shap.py` | `get_shap_explainer_and_values(model, X_train, X_test)`: SHAP explainer and SHAP values |
| `5-logisitc_regression.py` | `Logistic_Regression_Model(random_state)`: untrained `LogisticRegression` classifier |
| `6-svm.py` | `get_SVM_model(name, random_state)`: untrained `SVC` with a `linear`, `poly` or `rbf` kernel |

## Requirements

- Ubuntu 20.04 LTS, Python 3.11
- `pycodestyle` 2.14.0
- `numpy`, `matplotlib`, `scikit-learn`, `shap`
