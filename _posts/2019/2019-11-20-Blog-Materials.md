---
layout: post
title: Blog Materials
date: 2019-11-20
tags: machine_learning blog
categories: machine_learning blog
author: GaoangLau
---
* content
{:toc}


Boosting 与 bagging 区别联系
自举汇聚法(bootstrap aggregating)，也称为Bagging ，是从原始数据集选择S次后得到S个新数据集的一种技术。 




Boosting 是一种与 bagging很类似的技术，但在boosting中，不同的分类器是通过串行训练而获得的，每个新分类器都根据已经训练出来的分类器的性能来进行训练。boosting是通过集中关注被已有分类器错分的那些数据来获得新的分类器。 
Bagging中的分类器权重是相等的，而boosting并不相等，每个权重代表的是其对应分类器在上一轮迭代中的成功度。

Boosting代表 AdaBoost, XGBoost

## AdaBoost - 自适应Boosting
### Adaptive boosting 
运行过程: 训练数据中的每一个样本，并赋予其一个权重。首先在训练数据上训练出一个弱分类器并计算该分类器的错误率$$\epsilon = \frac{N_{\text{wrong}}}{N_\text{all}}$$，然后在同一数据集上再次训练弱分类器，在这次训练中，将会调整样本的权重，第一次分对的样本权重会降低，第一次分错的样本权重将会提高。

1. 基于单层决策树构建弱分类器。 

单层决策树(decision stump，也称决策树桩)是一棵只有一个根结点，两个叶子结点的简单决策树。 它是AdaBoost中最流行(并不是唯一)的弱分类器，


## CART 
CART, Classification And Regression Trees, 分类决策树。优点，可以对复杂和非线性的数据建模；缺点是，结果不易理解。

CART 采用二元切分来处理连续型变量，即每次把数据集切成两份，如果数据的某特征值大于切分所要求的值，那么这些数据就进入树的左子树，反之进入树的右子树。

如何度量连续型数值的不一致度？首先计算所有数据的均值，然后计算每条数据的值到均值的差值(绝对值或者平方值)。 

## Decision Trees 
* 计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可能处理不相关特征数据 
* 可能会过拟合


### ID3
Iternative Dichotomizer, the first of three Decision Tree implementations developed by Ross Quinlan (Quinlan, J. R. 1986.  Induction of Decision Trees. Mach. Learn. 1, 1 (Mar. 1986), 81-106.)

ID3的一般思路是：
1. 测量集合数据的熵
2. 寻找最优方案(特征)划分数据集
3. 对子集进行递归划分直到子集中所有数据属于同一个分类，或者特征耗尽

划分数据集的大原则是：**将无序的数据变得更加有序**。ID3使用**信息增益**(数据集划分前后信息发生的变化)的方法来划分。

要计算信息增益，我们需要一种度量集合信息的方式，比如香农熵(简称熵)。熵定义为信息的期望值，对于待分类的事物，符号$$x_i$$的信息定义为
$$l(x_i) = - \text{log}_2p(x_i)$$，其中$$p(x_i)$$为该分类的的概率。

由这些分类构成的集合的熵 $$H = -\Sigma_{i=1}^n p(x_i) \text{log}_2 p(x_i) $$ 。 从物理意义上直观的讲，熵对应一个系统的混乱与不一致程度，熵越大，表明这个系统越混乱。
信息增益刻画的是：熵的减少或者数据无序度的减少。 

### Gini impurity 
TODO

### Decision Tree Python Implementation 

[Deicision-Tree-ID3-Python3]({{site.baseurl}}/codes/decision_tree.py.txt)

ID3 的缺陷：
1. 数据集不够大时，很容易过拟合
2. 每次只能考察一个特征来作决策
3. 无数处理(连续)的数值特征及缺失值

### ID3 VS. C4.5
1. ID3 uses information gain whereas C4.5 uses gain ratio for splitting. 
2. ID3 每次划分分组时都会消耗特征，即划分数据分组之后特征数目会减少，而C4.5 & CART并不总是消耗特征
3. TODO

C4.5 over ID3
1. accepts both continuous and discrete features
2. handles incomplete data points; 
3. solves over-fitting problem by (very clever) bottom-up technique usually known as "pruning"; 
4. different weights can be applied the features that comprise the training data.



