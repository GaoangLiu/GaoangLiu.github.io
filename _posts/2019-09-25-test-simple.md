---
layout:     simple
title:      expor so 
date:       2019-09-25
tags: [Deep Learning, AI]
categories: 
- AI
- Deep learning
---

{%highlight python%}
def expr(str):
    print(str)
{%endhighlight%}    


<img src="dl.vs.ai.png">

Feedly is a news aggregator application for various web browsers and mobile devices running iOS and Android. It is also available as a cloud-based service. It compiles news feeds from a variety of online sources for the user to customize and share with others. Feedly was first released by DevHD in 2008. Wikipedia

Developer(s): DevHD

Operating system: Android 5.1 or later; iOS 10.0 or later (iPhone, iPad, and iPod touch)

Initial release date: 2008

License: Freemium

Platforms: Web browser, Handheld Devices

Written in: Java (back-end), JavaScript, HTML, Cascading Style Sheets (UI)

```
def expr(str):
    print(str)
```

Some formula:
$$ \alpha \times \beta = \Lambda $$

20世纪40~60年代，控制论(cybernetics)。随着生物学习理论的发展与第一个模型的实现(如感知机1958)，能实现单个神经元的训练。

<!--break-->

# 历史
$abc$


### 三次浪潮
20世纪40~60年代，控制论(cybernetics)。随着生物学习理论的发展与第一个模型的实现(如感知机1958)，能实现单个神经元的训练。

20世纪80~90年代(1980~1995)，联结主义(connectionism)。 可以使用反向传播(1986)训练具有一两个隐藏层的神经网络。

2006~now，深度学习的复兴

### Deep Learning v.s. AI

许多AI任务可以通过**提取一个合适特征，然后将这些特征提供给简单的机器学习算法**来解决问题。 但提取哪些特征是一个难题，解决这个问题的途径之一是**使用机器学习来发掘表示本身**，这种方法称为：表示学习(representation learning)。

典型例子：自编码器(autoencoder)。学习到的表示往往比手动设计的表示表现的要好，且只需极少人工干预。

表示学习的一个困难在于：多个变差因素同时影响着我们能够观察到的每一个数据，从原始数据是抽取高层次、抽象的特征非常困难。深度学习通过其他简单的表示来表达复杂表示，这解决了表示学习的核心问题。
典型例子：前馈深度网络。 

总：DL是ML的一种，是一种能够使用计算机系统从数据和经验中得到提升的技术。

<p align="center"> <img src="{{site.baseurl}}/images/dl.vs.ai.png" style="width:600"> </p>
<!-- ![some](/images/dl.vs.ai.png) -->



<!-- Math -->
# Basic Math

## 线性相关与生成子空间
<p align="center">  $$ Ax = b  $$ </p>
其中 $$A \in \mathrm{R}^{m \times n}$$ matrix, $$b \in \mathrm{R}^n $$ vector。对于上面的方程组来说，要么不存在解，要么存在唯一解或者无穷解，不可能存在大于1但小于无穷个解的情况 (不然，两个解的线性组合 $$\alpha x + (1-\alpha)y$$也是方程组的解)。

### 方程在每一点存在解的必要条件 $$ n \geq m$$

$$Ax = \Sigma_{i\in[1,n]} x_i A_{:, i} = \Sigma_i c_i v^{(i)}$$, where $$v^{(i)}$$ 是$$A$$的列向量。 判定以上方程组是否存在解，即判定$$b$$是否在$$A$$的生成子空间中。 这个特殊的生成子空间称为$$A$$的**列空间**或**值域(range)** . 

因为 $$ b \in \mathrm{R^m} $$，如果 $$\mathrm{R}^m$$ 中一个点不在 $$A$$ 的列空间中，那该点对应的 $$b$$ 没有解，因此 $$A$$ 至少有 $$m$$ 列，即 $$n \geq m$$。 举例，$$ m = 3, n = 2 $$，那么无论 $$x$$ 如何变化，它只能将 $$A$$ 映射到 $$\mathrm{R}^3 $$的一个平面，只有当 $$b$$ 处于这个平面时， 方程才有解。 

### 存在解的充分条件
$$ n \geq m$$ 并不能保证方程一定存在解，因为列向量可能**线性相关**，即某一个向量可能表示为其他一组向量的线性组合。要使其列空间涵盖整个 $$\mathrm{R}^m$$， 需要满足什么条件 ？

A: 矩阵必须包含**至少一组$$m$$个线性无关**的向量。

但要使矩阵可逆，需要保证对每一个 $$b$$ 至多只有一个解，因此要保证矩阵至多只有 $$m$$ 个列向量，即必须是一个方阵(square)，且所有列向量线性无关。这样的矩阵称为 **奇异矩阵**。

### 特征分解 
矩阵 $$A$$ 的**特征向量(eigenvector)**是指与 $$A$$ 相乘后相当于对该向量进行缩放的非零向量 $$v$$:  $$Av = \lambda v$$ ，
其中标题 $$\lambda$$ 称为这个特征向量对应的**特征值(eigenvalue)**. 
* 如果 $$v$$ 是 $$A$$ 的特征向量，那么任意缩放后的向量 $$sv(s \in \mathrm{R}, s \neq 0)$$ 也是 $$A$$ 的特征向量





 



