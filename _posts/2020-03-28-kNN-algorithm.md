---
layout:     post
title:      kNN algorithm
date:       2020-03-28
tags: [machine learning, kNN]
categories: 
- machine learning
---

## WHAT
1. **惰性学习模型**，也称**基于实例的学习模型**，对训练数据进行少量的处理或者不处理
2. 从样本数据集中选择前 $$k$$ 个最相似的数据，提取他们的分类标签；二分类中 $$k$$ 可以设置成奇数来避免平局现象
3. 邻居代表了 **度量空间** 的训练实例，度量空间定义了集合中所有成员之间距离的特征空间
4. 基于一个假设：互相接近的实例拥有类似的响应变量
5. 非参数模型，模型的参数不固定，可能随实例数量的增加而增加

## HOW 
1. 基于**特征相似度**, 按测量不同特征之间距离的方法来分类，将新数据的每个特征跟样本数据中的数据对应的特征比较
2. 如何选择 $$k$$，一个经验性的选择是 $$k = sqrt(N)$$，其中 $$N$$ 是样本个数

## WHEN
1. Dataset: **small, noise free, labelled**

## 特点
1. 精度高、对异常值不敏感
2. 时间、空间复杂度高(须保留全部数据集、须对数据集的每个数据计算距离值)


## Tools & Libraries 
### sklearn
`sklearn.neighbors` [source code](https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/neighbors/_classification.py#L25)
```python
sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)

```

Simple examples [https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

