---
layout:     post
title:      Boosting
date:       2020-07-22
tags: [boost, adaboost]
categories: 
- machine learning
---

作为实现 Ensemble 的途径之一，Boosting 是一族将弱学习器提升为强学习器的算法。这族算法的一般工作机制是：从初始训练集中训练出一个基学习器，再根据学习器的表现对训练样本的权重进行调整，基于调整后的样本再训练下一个学习器。训练过程以优化一个目标函数为指导，最终的模型将所有学习器进行加权组合。

Boosting 代表算法: AdaBoost[Yoav Freund \& Robert Schapire](https://bit.ly/3oQi5k4), Gradient Boost.

## Adaboost 思想
1. 提高被错误分类的样本的权值
2. 加权多数表决。加大分类误差率小的分类器的权重，减小误差率大的分类器的权重。 

<img src="http://git.io/JLTYM" alt="AdaBoost 伪代码" width="500px">

代价函数(指数损失函数) $$\mathcal{C} = E_{x \in D} \exp^{-f(x) H(x)} = \exp^{-f(x)\cdot \Sigma_j^T \alpha_t h_t(x)}$$。基分类器最常见的是决策树（深度为1的决策树）。

## Gradient Boosting 
Gradient Boosting = Gradient Descent + Boosting

和 AdaBoost 一样，Gradient Boosting 也是重复选择一个表现一般的模型并且基于先前模型的表现进行调整。不同点在于，AdaBoost 是通过**提升错分数据点的权重**来定位模型的不足而 Gradient Boosting 是通过**算梯度（gradient）**来定位模型的不足。因此相比 AdaBoost, Gradient Boosting 可以使用更多种类的目标函数。



## Adaboost, GBDT 与 XGBoost 的区别





## 参考

- [知乎: Adaboost, GBDT 与 XGBoost 的区别](https://zhuanlan.zhihu.com/p/42740654)
- [A Gentle Introduction to Gradient Boosting](https://www.chengli.io/tutorials/gradient_boosting.pdf)
  