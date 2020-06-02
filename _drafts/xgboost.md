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


# Sklearn XGBoost API 
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


