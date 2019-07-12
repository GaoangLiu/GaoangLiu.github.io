---
layout:     post
title:      机器学习笔记
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



# 监督学习

* 为什么要对数据进行归一化处理？
  * 做为一个特征，我们希望看到数据的相对值差别对结果的影响，而不是其绝对值。特别地，未做归一化的数据中，取值范围最大的数据将主导诸如kNN算法的结果
* 通俗解释过拟合、欠拟合？
  * 前者指一个模型过分关注训练数据，但对新数据的泛化性能不好，后者指模型无法获取数据中的所有变化。
* kNN算法优缺点？
  * 优势 精度高、对异常值不敏感、无数据输入假定 
  * 缺点 计算、空间复杂度高，无数给出数据的内在含义
* 决策树优缺点？
  * 计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特征数据
  * 得到的模型很容易可视化，非专家也很容易理解
  * 算法完全不受数据缩放的影响。每个特征被单独处理，数据的划分也不依赖于缩放，因此决策树算法不需要特征预处理，比如归一化或者标准化
  * 缺点 容易过拟合
* Entropy ?
  * 信息增益(information gain)，指对数据集进行处理之前之后发生的变化。对一个符号$$x$$ 的信息定义为 $$-\text{log}_2p(x)$$ 。这是一个 xxx 
  * 熵定义为信息的期望值 $$ H = - \Sigma_i^n p(x_i) \text{log}_2p(x_i)$$

### 线性回归

线性回归，也称普通最小二乘法(ordinary least squares, OLS)，回归问题最简单也最经典的线性方法。线性回归寻找参数 $$w$$与$$b$$，使得对训练集的预测值与真实的回归目标值 $$y$$之间的**均方误差**最小。

### 岭回归(ridge regression)

对于高维数据集(即有大量特征的数据集)，线性模型过拟合的可能性变大。 在岭回归中，对系数$$w$$的选择不仅要在训练数据上得到好的预测结果，还要**拟合附加约束**(E.g., 正则化)。 

Ridge模型在模型的简单性(系统都接近于0)与训练集性能之间做出权衡。



### 决策树

对数据反复进行递归划分，直到每个区域（叶结点）只包含单目标值（单一类别或单一回归值）。 

通常来说，构造决策树直到所有叶结点都是纯的，这会导致模型非常复杂，并且对训练数据高度拟合。典型的特征是：决策边界过于关注远离同类别样本的单个异常点。 这也是决策树的一个主要缺点之一。

防止过拟合：

1. 预剪枝(pre-pruning)：限制树的最大深度、叶结点的最大数目、规定一个结点中数据点的最少数据数目
2. 后剪枝(post-pruning) 先构造树，随后删除或折叠信息量很少的结点

决策树的优点： 

1. 

为了克服决策树过拟合的缺点，一个思路是合并多个决策树，即是构建：

### 随机森林

随机森林本质是：**多个决策树的组合**。 背后思想：每棵（决策）树的预测可能都相对较好，但可能对部分数据过拟合，如果构造很多树，并且每棵树的预测都很好，但以不同的方式过拟合，那么对这些树的预测结果取平均值来降低过拟合（对于分类问题，可以采用“软投票(soft voting)”策略，即每个算法做出“软”预测，给出每个可能输出label的概率，所有概率求平均值，输出概率最大标签）。

随机化方法

* 通过选择用于构造树的数据点，比如使用**自助采样(bootstrap sample)**, 从n_samples个数据点中有放回地随机抽取样本
* 通过选择每次划分测试的特征，每个树随机选择特征的一个子集。潜在问题 a. max_features 过大，比如等于n_features，那么所有树都考虑了全部特征，那么将十分相似 ; b. Max_features过小，比如1，为了更好拟合数据，每棵树都很深

随机森林也可以给出特征重要性(由所有树的特征重要性求和再平均)，一般来说，比单棵树给出的可为可靠。

#### 优、缺点

* 方法强大：通常不需要反复调节参数就可以得到很好的结果，也不需要对数据进行缩放。有决策树所有优点，也弥补了其过拟合的缺陷。但如果需要以可视化方式向非专家总结预测过程，选单个决策树可能更好。
* 支持多核并行， n_jobs = 9 or -1 
* 对于维度非常高的稀疏数据（比如文本数据），RF表现往往不是很好，线性模型可能更适合。

### 梯度提升机（梯度提升回归树）

采用连续的方式构造树，每棵树都试图**纠正前一棵树的错误**。优势，通常使用深度很小(1~5)的树，占用内存少，预测速度更快。

背后思想：**合并多个简单的模型(弱学习器)，比如深度较小的树。每棵树只能对部分数据做出好的预测，通过添加更多的树，不断迭代提高性能**。 

优点：

* 深度很小、占用内在少、预测速度很快 ; 表现很好
* 不需要数据缩放

缺点：

* 需要仔细调参，训练时间可能会比较长
* 不适用于高维稀疏数据

它对参数设置比rf更为敏感，如果设置得体，精度很高。故经常是ML竞赛优胜者。

### 核支持向量机

Kernelized support vector machine 通常简称 svm

### 神经网络

Q:  缺点？

* 功能越强大的神经网络，通常需要更长的训练时间; 还需要仔细的预处理数据
* 调参是一门艺术
* 在“均匀”的数据上表现良好，即特征都具有相似的含义。 如果数据包含不同种类的特征，那么基于树的模型可能表现的更好。 



# 无监督学习

* 如何评估无监督学习?
  * 通常来说，评估监督算法的唯一方法就是 **人工检查 **。 
* 





# CNN 卷积神经网络

CNN(convolutional neural networks) 

CNN的基石是卷积层(convolutional layer)。Input layer的接收野(receptive fields, 一个$$m \times n$$ 像素矩阵)连接第一卷积层的一个神经元，同样，第1卷积层的一个 $$k \times l $$ 神经元矩阵连接到第2层卷积层。因此，第$$i$$层卷积层的第 $$j, j+1$$个神经元所对应的第 $$i-1$$ 层神经元矩阵可能是重叠的($$k = l = 1$$就不重叠)。

有时候为了使第 $$i, i+1$$ 卷积层长宽一致，会在卷积层周围添加一些 0, 称为**0填充**. 

[<p align="center"><img src="img/zeropadding.png" width="500"/></p>](zero padding)





![ks](zeropadding.png)





### Messy notes

* Octave lib (Prof. Andrew Ng 建议初学者使用)˘˘

˘˘