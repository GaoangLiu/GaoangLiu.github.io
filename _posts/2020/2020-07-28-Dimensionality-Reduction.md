---
layout: post
title: Dimensionality Reduction
date: 2020-07-28
tags: machine_learning pca
categories: machine_learning
author: GaoangLiu
---
* content
{:toc}

降维就是一种对高维度特征数据预处理方法。降维是将高维度的数据保留下最重要的一些特征，去除噪声和不重要的特征，从而实现提升数据处理速度的目的。



降维方法：投影、流形学习
主流技术: PCA, Kernel PCA, 局部线性嵌入(LLE)。


降维具有如下一些优点：
1. 使得数据集更易使用。
2. 降低算法的计算开销。
3. 去除噪声。
4. 使得结果容易理解

## PCA
在信号处理领域，我们认为信号具有较大方差，噪音具有较小方差，信号与噪音之比称为信噪比。比值越大意味着数据的质量越好，反之越差。
PCA 的思路有多种，比如：
1. 最大化投影方差
2. 最小回归误差

### 最大化投影方差
对于给定样本集 $$ \{ u_1, ..., u_n \} $$，其中每个 $$ u_i $$为一个列向量，表示一个样本。再假设$$ V = \{ v_1, ..., v_n \} $$为中心化处理的样本集，即么有 $$ \sum\limits_{i=1}^n v_i = 0 $$. 

令 $$ \omega $$ (单位向量)表示一新的坐标轴方向，则 $$ v_i $$ 在 $$ \omega $$ 上的投影为 $$ (v_i, \omega) = v_i^T \omega $$。则样本集投影后的方差为 
$$
\begin{aligned}
 E(v_i^T \omega)^2 - (E(v_i^T \omega))^2 \\ 
 = \frac{1}{n} \sum\limits_{i=1}^n \omega^T v_i v_i^T \omega \\
 = \omega^T(\frac{1}{n}\sum\limits_{i=1}^n v_i v_i^T) \omega
\end{aligned}
$$
令 $$ C = \frac{1}{n}\sum\limits_{i=1}^n v_i v_i^T $$， 最大化投影后的方差则则转化为一个最大化问题：
$$
\begin{aligned}
\text{max}(\omega^T (\frac{1}{n}\sum\limits_{i=1}^n v_i v_i^T) \omega) \\
\text {s.t., } \omega^T \omega = 1
\end{aligned}
$$

引入拉格朗日乘子 $$ \lambda $$,  并对 $$ \omega $$ 求导，可以得到 $$ C \omega = \lambda  \omega $$。可见向量 $$ V $$ 投影后的方差即是协方差的特征值。因此最大方差也对应着协方差矩阵最大特征值，新的坐标向量即最大特征值对应的特征向量。次投影方向是第 2 大特征值对应的特征向量。 

通过这最大投影方法进行 PCA 的一般步骤:
1. 对样本数据进行中心化处理
2. 求样本协方差矩阵
3. 对协方差矩阵进行特征值分解
4. 取前 $$k$$ 个最大特征值对应的特征向量 $$\omega_1, ..., \omega_k$$，能过以下映射将 $$ n $$ 维样本映射到 $$k$$ 维 $$ v_i' = (\omega_1^T v_i, ..., \omega_k^T v_i)^T $$

### 最小平方误差
找到一个 $$ k $$ 维超平面，使得数据点到这个超平面的距离平方和最小。
样本点 $$ x_i $$ 到 $$ k $$ 维超平面的距离为 $$ (x_i, D) = \| x_i - x_i'\|$$，其中 $$ x_i' $$ 表示 $$ x_i $$ 在$$ D $$ 上的投影向量。假设 $$D$$ 由正交基 $$ W = \{ \omega_1, ..., \omega_k\}$$，则 $$ x_i' = \sum\limits_{i=1}^k(\omega_i^T x_i) \omega_i $$。因此 PCA 要优化的目标:

$$
\begin{aligned}
    \underset{\omega_1, ..., \omega_d}{\text{arg min}} \sum\limits_{i=1}^n \| x_i - x_i' \|^2
\end{aligned}
$$


### PCA 小结
PCA 对数据缩放比较敏感，


### PCA 小结

缺点
1. 主成分各个特征维度的含义比较难以解释

