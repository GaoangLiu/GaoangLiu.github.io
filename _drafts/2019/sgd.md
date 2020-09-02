---
layout:     post
title:      Gradient Descent
date:       2019-10-20
tags: [machine learning]
categories: 
- machine learning
---

## 梯度(Gradient)
梯度是张量运算的导数，是导数这一概念向多元函数导数的推广。

GD的优势：简单、有效，对于凸函数(convex)来说，GD总能很快找到最小值。 相应的，对于非凸函数(non-convex)，GD可能会陷入到一个局部最小值，而无法收敛到全局最小值。 

为了使网络收敛，改进方案: **改变学习速率** 、 **use momentum**

## RMSprop Optimizer


## SGD
缺点：如果函数的形状非均向(anisotropic)，比如呈延伸状，那么寻找最小值的路径将非常低效。其根本原因在于，梯度的方向并没有指向最小值的方向。

## 小批量随机梯度下降(mini-batch stochastic gradient descent)
思想：从样本中抽取训练样本$$X$$和对应目标，然后在$$X$$上运行网络。
