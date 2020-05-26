---
layout:     post
title:      Naive Bayes Classifier
date:       2020-04-06
tags: [Bayes, Machine Learning]
categories: 
- Machine Learning
---

# Bayes
## Multinomial Naive Bayes 
`sklearn.naive_bayes.MultinomialNB(*, alpha=1.0, fit_prior=True, class_prior=None)`


## What ?
* 核心思想：选择具有最高概率的决策
* 基于两个假设
    1. 特征相互独立
    2. 特征同等重要 

## When? 
在实际使用情况下，NB classifier 通常表现不错，特别适用于文档分类及垃圾信息过滤。也可以作为一个很好的基准模型。 

### Pros
* 非常适合于分类问题，速度比线性模型快
* 适用于非常大的及高维数据集合，精度通常低于线性模型

高效的原因，NB 通过单独查看每个特征来学习参数，并从每个特征中收集简单的类别统计数据。

## Scikit-Learn module
The [sklearn module](https://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes) implements the following mothods:

```python
from sklearn.naive_bayes import MultinomialNB, GaussianNB, \
    BernoulliNB, ComplementNB, CategoricalNB
```

## Implementation
简单 NB classifier 实现 [bayes.py]({{site.baseurl}}/codes/2020/bayes.py).

具体测试结果与 `sklearn.naive_bayes.MultinomialNB` 效果一致，对[垃圾邮件](https://www.dropbox.com/s/yjiplngoa430rid)筛选处理，都达到了以下结果

```bash
==  using 2000 features, confusion matrix
[[129   9]
 [  1 121]]
Accracy score:   0.9615384615384616
```

# Reference 
* [Why naive Bayes works well](https://www.cs.unb.ca/~hzhang/publications/FLAIRS04ZhangH.pdf)

## U2B Videos
* [Bayes theorem, and making probability intuitive](https://www.youtube.com/watch?v=HZGCoVF3YvM)
