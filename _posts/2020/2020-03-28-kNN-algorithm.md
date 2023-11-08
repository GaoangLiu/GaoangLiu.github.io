---
layout: post
title: kNN algorithm
date: 2020-03-28
tags: machine_learning knn
categories: machine_learning
author: berrysleaf
---
* content
{:toc}




## WHAT
1. **惰性学习模型**，也称**基于实例的学习模型**，对训练数据进行少量的处理或者不处理



    - 称其`懒惰`因为这个算法并没有去学习一个判别式函数(损失函数)，而是记住整个训练集

2. 从样本数据集中选择前 $$k$$ 个最相似的数据，提取他们的分类标签；二分类中 $$k$$ 可以设置成奇数来避免平局现象
3. 邻居代表了 **度量空间** 的训练实例，度量空间定义了集合中所有成员之间距离的特征空间
4. 基于一个假设：互相接近的实例拥有类似的响应变量
5. 非参数模型，模型的参数不固定，可能随实例数量的增加而增加


## HOW 
1. 基于**特征相似度**, 按测量不同特征之间距离的方法来分类，将新数据的每个特征跟样本数据中的数据对应的特征比较
2. 如何选择 $$k$$，一个经验性的选择是 $$ k = \sqrt{N} $$，其中 $$N$$ 是样本个数

## 特点
1. 优点
    * 对异常值不敏感
    * 模型容易理解，通常不需要过多调节就可以得到不错的性能，适合处理多分类问题，比如推荐用户
    * 可用于数值型及离散型数据，可分类可回归
2. 缺点
    * 时间、空间复杂度高(须保留全部数据集、须对数据集的每个数据计算距离值)
    * 需要对数据进行缩放[Data scaling]/标准化({{site.baseurl}}/archives/Processing-data-with-Python.html)，不然方差大的特征将主导样本点之间距离
    * 对于特征多的数据集往往效果不好，对于稀疏数据集(大多数特征的大多数取值都为 0 的数据集)效果尤为不好
    * 向量维度越高，欧氏距离区分能力越弱 


## 对比 K-Means

* KNN 属于监督学习算法，K-means 属于非监督学习算法；
* KNN 没有前期训练过程，K-means 有训练过程；
* KNN 属于分类算法，目的是为了确定一个点的类别。K-means 属于聚类算法，目的是为了将一系列点集分成k类；
* 相似点：都用到了 NN(Nears Neighbor) 算法来寻找给定点距离最近的点，一般用 KD 树来实现 NN； 


## Tools & Libraries 
### sklearn
`sklearn.neighbors` [source code](https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/neighbors/_classification.py#L25)
```python
sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)
```

Simple examples [https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

-- [Simple example.1 of kNN](https://bit.ly/32S9GVA) --

# References

[Benchmarking Nearest Neighbor Searches in Python](https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/#Scaling-with-Leaf-Size)


