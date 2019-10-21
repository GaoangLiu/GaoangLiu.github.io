---
layout:     post
title:      Gradient Descent
date:       2019-10-20
tags: [Macgine Learning, gd]
categories: 
- Machine Learning
---

## 梯度(Gradient)
梯度是张量运算的导数，是导数这一概念向多元函数导数的推广。


## SGD
缺点：如果函数的形状非均向(anisotropic)，比如呈延伸状，那么寻找最小值的路径将非常低效。其根本原因在于，梯度的方向并没有指向最小值的方向。

## 小批量随机梯度下降(mini-batch stochastic gradient descent)
思想：从样本中抽取训练样本$$X$$和对应目标，然后在$$X$$上运行网络。
