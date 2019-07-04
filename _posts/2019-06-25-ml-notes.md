---
layout:     post
title:      Machine Learning notes
subtitle:   
date:       2019-06-25
author:     ssrzz
catalog: 	true
tags:
  - ml
  - tutorial
---

## Define ML

- Arthur Samule(1959): fields of study that gives computers ability to learn without being explicitly programmed. 

  > Samule 曾写了个跳棋(checker)程序，不断训练让程序学习什么是好的move，什么是坏的。尽管Samule本人并不擅长checker，但一个不擅长checker的人在1950s能编写并训练出一个比自己更擅长checker的程序，这是一件很了不起的事情。

* Tom Mitchell(1998): A computer program is said to learn some experience $$E$$ w.r.t. some task $$T$$ and some performance $$P$$, if its performance $$P$$, as measured by $$T$$, improves with experience $$E$$. 

  i.e., 关于任务$$T$$的程序$$Pr$$ 的成绩$$P$$， 正比于从$$T$$中学习到的经验$$E$$. 

  $$P(Pr_T) \propto E(Pr_T)$$

  > 上述定义其实是“针对” good ML program，如果一个算法越学越差，本质上也是一个ML算法，但只能算是一个糟糕的算法 (分类问题+准确率极低的算法+结果取反 = 一个准确率还不错的算法？)。



### Quick question: what do you think about AI?

AI的本质是, 人类所赋予机器(or 工具)的一种能够解决具备一定难度的问题的能力(the ability to solve certain difficult problems — difficult in the sense that they are not easily solved by humans)，这种能力越强，就能越好的解决问题(e.g., 准确率更高的分类器)。 

但机器所被赋予的intelligence，在相当长一段时间内都是比较单一、针对一个或者一类具对问题的能力，它们很难做到像人类一样进行逻辑推理、反思总结并把结果推广到其他问题上。未来的技术有可能创造出[终结者](https://zh.wikipedia.org/zh-hans/终结者)这样的AI — 目的单一：杀死Sara，阻碍John Connor的诞生(其用到的人脸识别、运动中识别并跨越、躲避障碍物的技术都将在未来数年内得到大幅提升)。但行动自然、可以熟练运用计谋、同情心、美色诱惑并欺骗人类以逃离的[机械姬 Ex Machina](https://zh.wikipedia.org/zh-hans/机械姬)则不太可能。

AI很难做出人类的复本，但也可能完全没有这个必要 。飞机像鸟类一样飞翔，但不必振动机翼。

### 问题分类

1. regression problem v.s. classification
   * 主要区别：预测结果是连续的(continuous valued output)还是离散的(discrete)。问题可以相互转化，二者界线也并非那么绝对。房价是连续的，但可以划分为离散的区间 (A: 0-10Million, B:10~20M, C:...)

2. supervised v.s. unsupervised
   * supervised: each example has a label ("right" answer), we teach program how to learn
   * unsupervised



## Linear Regression

### Hypothesis $$h$$

$$h_\theta(x^i) = \theta \cdot x^i =  \theta_0 + \theta_1 x_1^i + \cdots + \theta_n x _n^i$$

###Why prediction function is called hypothesis ?

历史残留原因，早期ml领域采用了这个术语，一直沿袭至今。



#### Parameters $$\theta = (\theta_0, …, \theta_n)$$

#### Goal

代价函数$$J(\theta) = \frac{1}{2m} \mid h_\theta(X) - Y \mid ^2 =\frac{1}{2m} \Sigma_{i=1}^m \mid h(x^i) - y^i \mid ^2$$, 找到 $$\theta$$ 使得 $$\text{minimize}_{\theta} J(\theta) $$

### Why square error function?

对于多数regression问题来说，sef 都是一个不错的选择





## GD(gradient descent)

### "Batch" GD

batch : Each step of GD uses all the training examples

​	$$ \text{repeat} \{ \theta_j = \theta_j - \frac{\partial}{ \partial \theta_j} J(\theta) \}  = \theta_j - \alpha/m \Sigma_1^m(h_\theta(x^{(i)} -y^{(i)} ) x_j^{(i)} )$$

### Feature scaling 

* why ? 从以上GD的迭代过程看到，**$$\theta_j$$的更新幅度与$$x_j$$相关**。如果某个$$x_j$$的尺度相较其他维度特征大很多，势必造成该维度对应的参数$$\theta_j$$的更新非常剧烈，而其他维度特征对应的参数更新相对缓和，这样造成迭代过程中很多轮次实际上是为了消除特征尺度上的不一致，而不是快速的找到最优点，最终极大影响收敛速度。

* How ? E.g., mean normalization (均值标准化？)

  $$x_i  = \frac{x_i - u_i}{s_i}$$，其中  $$u_i = 1/\mid x_i\mid \sum(x_i^j), s_i = \text{max}(x_i) - \text{min}(x_i)$$



## Logistic Regression 

### Cost function

$$J(\theta)  = \frac{1}{m} \sum C(h_\theta(x^{(i)}), y^{(i)})$$

$$ C(h_\theta(x), y) = -log(h_\theta(x)) \text{  if  } y == 1 \text{ else } -log(1 - h_\theta(x)) $$

因为$$y$$总是取0 or 1， 上式可改写为 $$C(h_\theta(x), y)  = - y log(h_\theta(x))  - (1-y) log(1 - h_\theta(x))$$

直觉，如果假设与真实值相差很大，e.g., $$y = 1 , h_\theta(x) = 0$$ 那么代价函数也将极大。

#### Qestion: 为何选择这个CF?

原则上，CF是度量假设$$h$$与真实值$$y$$差异的函数。 一个好的CF应该xxx，并且能很好的指导如果去调整参数$$\theta$$. 如果使用linear regression中的代价函数是non-convex函数，





## Overfitting 

1. 减少特征数目 / reduce number of features
   * 手动选择重要的features
   * model selection algorithms 会自动选择相关的特征
2. 正则化
   * 保留所有features，但减小$$\theta_j$$的值



## System design

### 结果标准

* 正确率(precision)  $$P = \text{TP} / (\text{TP} + \text{FP})$$ ，即所有预测为1的示例中真实结果为1的概率. E.g., 97(real spam) / 100 predicted spam、98 (real cancer) / 101(predicted cancer)

* 召回率(recall) $$R=\text{TP}/(\text{TP} + \text{FN})$$，即真实结果为positive的示例被正确预测出来的概率. E.g., 97(real spam) / 100 (total spam)  , 98 (real cancer) / 101 (total cancer)

  precision / recall 不可兼得。

* $$\mathcal{F}_1$$ Score ($$\mathcal{F}$$ score) = $$2\frac{PR}{P+R}$$



### 正则

$$J(\theta) =\frac{1}{2m} [\Sigma_{i=1}^m \mid h(x^i) - y^i \mid ^2  + 	\lambda \Sigma_j^n\theta_n^2] $$,





## 推荐系统(recommender systems)



### Messy notes

* Octave lib (Prof. Andrew Ng 建议初学者使用)˘˘

˘˘