---
layout:     post
title:      Xgboost
date:       2020-06-01
tags: [xgboost]
categories: 
- machine learning
---

## What is `XGBoost`
XGBoost stands for **Extreme Gradient Boosting**, where the term "Gradient Boosting" originates from the paper *Greedy Function Approximation: A Gradient Boosting Machine*, by Friedman.

XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.

## Pros and Cons
Cons:
1. Compared with bagging (random froests), boosted trees are more susceptible to outliers, because it tries to fit to the error in each iteration. Bagging are robust against outliers, since each tree is built independently.

# Core Data Structure
`xgboost.DMatrix(data, label=None, weight=None, base_margin=None, missing=None, silent=False, feature_names=None, feature_types=None, nthread=None)`

DMatrix is a internal data structure that used by XGBoost which is optimized for both **memory efficiency and training speed**. 

Example on how to construct `DMatrix` from pandas DataFrame:
```python
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=89)
dtrain = xgboost.DMatrix(data=X_train.values, feature_names=X_train.columns, label=y_train.values)
```



# Sklearn XGBoost API 
Scikit-Learn Wrapper interface for XGBoost.
`class xgboost.XGBRegressor(objective='reg:squarederror', **kwargs)`:
Parameters:
* `n_estimators` (int), number of gradient boosted trees. Equivalent to number of boosting rounds
* `max_depth`, maximum tree depth for base learners
* `learning_rate`[0, 1], boosting learning rate(xgb's `eta`),
* `verbosity`, 0 (silent) - 3(debug)
* `objective` (string or callable), specify the learning task and the corresponding learning objective or a custom objective function to be used, e.g., `params = {'objective':'binary:logistic', 'n_estimators':2}`
* `n_jobs`, number of parallel threads used to run xgboost
* ...




## Visualize Gradient Boosting Decision Trees
```python
from xgboost import plot_tree

fig, ax = plt.subplots(figsize=(30, 30))
plot_tree(model, num_trees=4, ax=ax) # after training
```

## Early stopping 
```python
model.fit(X_train, y_train, early_stopping_rounds=10, 
            eval_metric="logloss", eval_set=eval_set, 
            verbose=True
```

## Save & load model 
```python
import pickle
pickle.dump(model, open('best_model.pkl', 'wb))

model = pickle.load(open('best_model.pkl', 'rb'))
```
Note that, if you save the model with one version of `xgboost` (e.g., 0.9) and load the model with another version, there might be errors. 


# Learning API
Default setting:
`xgboost.train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None, maximize=False, early_stopping_rounds=None, evals_result=None, verbose_eval=True, xgb_model=None, callbacks=None)`

Parameters:
* `num_boost_round`, number of boosting iterations, 
* `evals` (list of pairs (DMatrix, string)), list of validation sets for which metrics will evaluated during training. Validation metrics help tracking the performance of the model,
* `obj` (function), customized objective function,
* `feval` (function), customized evaluation function,
* `early_stopping_rounds` (int), activates early stopping. Note that, 
    1. training stops if no improvement was found on evaluation metric in an `early_stoppping_rounds` rounds. Then the method returns the model from the **last iteration (not the best one)**.
    2. The last entry in `evals`, if there are multiple entries, will be used for early stopping. 
    <!-- Therefor `evals=[(dvalid, 'valid'), (dtrain, 'train')]` -->
* `verbose_eval` (bool or int), set it to a number, e.g., 10 to print the evaluation metric result at every 10 boosting stage.
* `callbacks`, ...

Built-in cross validation: 

`xgboost.cv(params, dtrain, num_boost_round=10, nfold=3, stratified=False, folds=None, metrics=(), obj=None, feval=None, maximize=False, early_stopping_rounds=None, fpreproc=None, as_pandas=True, verbose_eval=None, show_stdv=True, seed=0, callbacks=None, shuffle=True)`:

Parameters:
* `nfold`, number of folds in CV, 
* `stratified` (bool), perform stratified sampling.


# Python examples
More examples on how to customized loss functions, cross validation, access evals result and etc. can be found on their [Github repo](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/basic_walkthrough.py).
