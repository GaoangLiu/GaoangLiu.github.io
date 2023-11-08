---
layout: post
title: Iterative Imputation for Missing Values
date: 2020-06-05
tags: imputer missing-value
categories: machine_learning
author: berrysleaf
---
* content
{:toc}


# Horse Colic Dataset
[Horse colic](https://www.kaggle.com/uciml/horse-colic?select=horse.csv) : to predict whether or not a horse can survive based upon past medical conditions.




A numeric version (categoric features transferred to numbers) of this dataset can be found [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv).

<img src="https://i.loli.net/2020/06/05/8tuPZ9oOUMW3aIJ.png" width="350px" class='center'>

There are 300 rows and 28 input variables with one output variable. It is a binary classification prediction task that involves predicting 1 if the horse lived and 2 if the horse died.

## Data exploration 
```
df = pd.read_csv('data.csv', names=cols, na_values='?')
df
```
shows
1. Most of the columns have missing values
2. columns `nasogastric_reflux_ph` (missing 247 (82.33%)), `abdomo_protein`(missing 198 (66.00%))
 and `abdomo_appearance`(missing 165 (55.00%)) have lost more than half of their values.

<img src="https://i.loli.net/2020/06/05/8kgnLtUaMJ6zFTs.png" width="750px" class='center'>

<img src="https://i.loli.net/2020/06/05/SgjrBylYvXQ2KF3.png" width="450px" class='center' alt='Missing value summary'>


# Iterative Imputation 
Iterative imputation refers to a process where **each feature is modeled as a function of the other features**, e.g. a regression problem where missing values are predicted. Each feature is imputed sequentially, one after the other, allowing prior imputed values to be used as part of a model in predicting subsequent features.

It is iterative because this process is repeated multiple times, allowing ever improved estimates of missing values to be calculated as missing values across all features are estimated.

Iterative imputer can do a number of different things depending upon how you configure it. Take the following values for example: 

```python
df = pd.DataFrame({'A':[np.nan,2,3], 'B':[3,5,np.nan], 'C':[0,np.nan,3], 'D':[2,6,3]})
"""
	A	B	C	D
0	NaN	3.0	0.0	2
1	2.0	5.0	NaN	6
2	3.0	NaN	3.0	3
"""
```

1. Imputer first fill the missing values based on the `initial_strategy` parameter (`default: mean()`)

```python
"""
	A	B	C	D
0	2.5 3.0	0.0	2
1	2.0	5.0	1.5	6
2	3.0	4.0	3.0	3
"""
```

2. Secondly it trains the estimator passed in [(`default = Bayesian_ridge`)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html) as a predictor. In our case we have columns: `A,B,C,D`. So the classifier would fit a model with independent variables `A, B, C` and dependent variable `D`, i.e., 
```python
X = df[['A','B','C']]
y = df[['D']]
model = BaysianRidge.fit(X,y)
```
The newly filled values will be flagged as `imputed`, and are then replaced by the predicted values of this model (other values will stay the same).
```python
df[df[D = 'imputed_mask']] = model.predict(df[df[D = 'imputed_mask']])
```

This method is repeated for all combinations of columns (the round robin described in the docs) e.g.
```python
X = df[['B','C','D']]
y = df[['A']]
...

X = df[['A','C','D']]
y = df[['B']]
...   

X = df[['A','B','D']]
y = df[['C']]    
...
```
This round robin of training an estimator on each combination of columns makes up **one pass**. This process is repeated until either the stopping tolerance is met or until the iterator reaches the max number of iterations(`default = 10`).


# Experiments
The scikit-learn library provides the `IterativeImputer` class that supports iterative imputation.
To use this tool, we have to imported two statements, the first one `enable_iterative_imputer` is required to enable the second module `IterativeImputer`.

By default, a [BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html) model is employed, features are filled in *ascending order*, from those with the fewest missing values to those with the most. 

The library also provides other strategies, such as *descending, right-to-left (Arabic), left-to-right (Roman), and random*.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(estimator=BayesianRidge(), n_nearest_features=None, imputation_order='ascending')
```

For this experiment, we have adopted:
1. `xgboost.XGBclassifier` with default parameters
2. 10-fold `RepeatedStratifiedKFold` for cross validation 

The mean accuracy of our model is `0.831`. For records, we have also tried using `mean()` and `mode()` method (with the same settings) to fill the missing values, both strategies outperform `imputer`. For example, with `mean()` we have acquired a mean accuracy score `0.848`.

```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

model = xgboost.XGBClassifier()
pipeline = Pipeline(steps=[('i', imputer), ('m', model)])
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)

scores = cross_val_score(pipeline, df, labels, scoring='accuracy', cv=cv, 
                        n_jobs=-1, error_score='raise', verbose=1)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
"""
Mean Accuracy: 0.831 (0.048)
"""
```


