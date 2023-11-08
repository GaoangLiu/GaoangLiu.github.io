---
layout:     post
title:      CatBoost
date:       2020-06-08
tags: [catboost, boost]
categories: 
- machine learning
---

# About
[CatBoost](https://catboost.ai/) is an algorithm for **gradient boosting on decision trees**. It is developed by Yandex researchers and engineers, and is used for search, recommendation systems, personal assistant, self-driving cars, weather prediction and many other tasks at Yandex and in other companies, including CERN, Cloudflare, Careem taxi. It is in open-source and can be used by anyone.


# Quick start
## CatBoostClassifier
```python
import numpy as np
from catboost import CatBoostClassifier, Pool

# initialize data
train_data = np.random.randint(0, 100, size=(100, 10))
train_labels = np.random.randint(0, 2, size=(100))
test_data = catboost_pool = Pool(train_data, train_labels)

model = CatBoostClassifier(iterations=2, depth=2, learning_rate=1, loss_function='Logloss', verbose=True)

# train the model
model.fit(train_data, train_labels)

# make the prediction using the resulting model
preds_class = model.predict(test_data)
preds_proba = model.predict_proba(test_data)
```