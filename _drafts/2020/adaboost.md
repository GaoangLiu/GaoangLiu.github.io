---
layout:     post
title:      AdaBoost Algorithm
date:       2020-06-11
tags: [adaboost]
categories: 
- machine learning
---

AdaBoost, stands for adaptive boosting. It is adaptive in the sense that **subsequent weak learners are tweaked in favor of those instances misclassified by previous classifiers**.

The core principle of AdaBoost is to **fit a sequence of weak learners (i.e., models that are only slightly better than random guessing, such as small decision trees) on repeatedly modified versions of the data**. The predictions from all of those weak learners are then combined through a weighted majority vote (or sum) to produce the final prediction.

Basic steps: 
* all training samples are assigned a weight $$\omega_i = 1/N$$
* a weak learner train on the original data
* sample weights are individually modified based on whether they're been correctly predicted. As iterations proceed, examples that are difficult to predict receive ever-increasing influence
* weak learners train on re-weighted samples, and repeat

---

# Implementation 

[AdaBoost Classifier Python Implementation](https://github.com/berrysleaf/berrysleaf.github.io/blob/master/codes/mlmodels/adaboost.ipynb)

---

# [sklearn API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)

`class sklearn.ensemble.AdaBoostClassifier(base_estimator=None, *, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)[source]`, default parameters:
* `n_estimators=50`, number of estimators 
* `learning_rate=1.0`, learning rate shrinks the contribution of each classifier 
* `algorithm=SAMME.R`, can be either `SAMME` (Stagewise Additive Modelling using a Multiclass Exponential loss function) or `SAMME.R` (R stands for real). If `SAMME.R` then use the `SAMME.R` real boosting algorithm. `base_estimator` must support calculation of class probabilities. If `SAMME` then use the SAMME discrete boosting algorithm. The `SAMME.R` algorithm typically **converges faster** than `SAMME`, achieving a lower test error with fewer boosting iterations.

# Boosting v.s. Bagging 
* Bagging, 
    1. often considers homogeneous weak learners, learns them **independently** from each other in parallel and combines them following some kind of deterministic averaging process
    2. focus at getting an ensemble model with less variance than its components, i.e., more robust model

* Boosting (bootstrap aggregating), 
    1. learns them sequentially in a very adaptive way (a base model depends on the previous ones) and combines them following a deterministic strategy
    2. try to produce **strong models** less biased than their components (even if variance can also be reduced)

|  | Bagging | Boosting |
|:---|:---|:----|
| Partitioning of data | Random | Higher vote to misclassified samples |
| Goal to achieve | Minimum variance | Increase accuracy | 
| Methods used | Random subspace | Gradient descent | 
| Functions to combine single model | Weighted average | Weighted majority vote  | 
| Example | Random Forest | Ada Boost | 
