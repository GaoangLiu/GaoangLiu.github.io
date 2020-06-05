import numpy as np
import pandas as pd 

from scipy.stats import uniform, randint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

import xgboost as xgb 

def classifier():
    model = xgb.XGBClassifier()
    model.fit(train, y)


def regressor():
    xgbreg = xgb.XGBRegressor(learning_rate=0.01, n_estimators=4460,
                       max_depth=5, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror', 
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006, verbose=True)
    folds = 10
    kfolds = KFold(n_splits=folds, shuffle=True, random_state=42)
    preds = np.zeros([test.shape[0],])

    for train_idx, test_idx in kfolds.split(X):
        X_train, X_val, y_train, y_val = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
        model = xgbreg
        model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_val, y_val)],
                early_stopping_rounds=10, eval_metric='rmse', verbose=0)

        y_pred = model.predict(X_val)
        print("RMSE is:", math.sqrt(sklearn.metrics.mean_squared_error(y_val, y_pred)))
        preds += model.predict(test)

    return preds / folds


def grid_search():
    # A parameter grid for XGBoost
    params = {'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [3, 4, 5] }

    model = xgboost.XGBRegressor(learning_rate=0.02, n_estimators=3460,
                        objective='reg:squarederror', nthread=-1,
                        scale_pos_weight=1, seed=27,
                        reg_alpha=0.00006, verbose=True)

    folds = 3
    kfolds = KFold(n_splits=folds, shuffle=True, random_state=42)

    # How many different combinations should be picked randomly out of our total (e.g., 405, see above). 
    param_comb = 5
    random_search = RandomizedSearchCV(model, param_distributions=params, 
                                    n_iter=param_comb, scoring='neg_mean_squared_error',
                                    n_jobs=-1, cv=kfolds.split(X, y), verbose=3, random_state=1001)

    random_search.fit(X, y)

    print('All results:')
    print(random_search.cv_results_)
    print('Best estimator:')
    print(random_search.best_estimator_)
    print('Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
    print(random_search.best_score_ * 2 - 1)
    print('Best hyperparameters:')
    print(random_search.best_params_)
    results = pd.DataFrame(random_search.cv_results_)
    
    return results