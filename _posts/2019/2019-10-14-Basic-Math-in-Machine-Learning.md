---
layout: post
title: Basic Math in Machine Learning
date: 2019-10-14
tags: machine_learning math
categories: machine_learning
author: berrysleaf
---
* content
{:toc}


# 奇异值分解
[定义] 将一个非零的 $$m \times n$$ 实矩阵 $$A$$ 表示为以下矩阵乘积形式的运算，即进行矩阵的因子分解: 



$$
\begin{aligned}
    A = U \Sigma V^T
\end{aligned}
$$
其中 $$U$$ 为 $$m$$ 阶正交矩阵， $$V$$ 为 $$n$$ 阶正交矩阵， $$\Sigma$$ 为对角矩阵，且对角元素 $$\lambda_1 \geq \lambda_2 \geq ... \lambda_p \geq 0$$。
则 $$U \Sigma V^T$$ 称为 $$A$$ 的奇异值分解(singular value decomposition, SVD), $$\sigma_i$$ 为奇异值。

[定理] 设 $$A \in \mathrm{R}^{m \times n}$$ 为一实矩阵，则 $$A$$ 的奇异值分解必须存在。


# 随机变量及分布

## 正态分布

若随机变量$$X$$的密度函数为

$$p(x) = \frac{1}{\sqrt{2\pi} \sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}, -\infty \le x \le \infty$$

则称 $$X$$ 服从正态分布，记作 $$X \sim N(\mu, \sigma^2)$$。其分布函数 

$$F(x) = \frac{1}{\sqrt{2\pi}\sigma} \int_{- \infty}^x  e^{-\frac{(t-\mu)^2}{2\sigma^2}} dt$$

正态分布由两个参数决定 $$\mu, \sigma$$：$$\mu$$决定密度函数的位置，称为位置参数； $$\sigma$$决定函数的尺度，称为尺度函数。

特别的，当$$\mu=0, \sigma=1$$的正态分布$$N(0, 1)$$为标准正态分布。 

### 一般正态分布的标准化

定理：若$$X \sim N(\mu, \sigma^2)$$，则$$U = (X - \mu) / \sigma \sim N(0, 1)$$

这个定理表明一般正态分布都可以通过一个线性变换化成标准正态分布。 

### 正态分布期望与方差
标准正态分布$$U$$的期望为 $$ E(U) = 1/\sqrt{2\pi} \int_{- \infty}^{\infty} u e^{-\frac{u^2}{2}} du$$，可以看到被积函数是一个奇函数(i.e., $$\forall x \in \mathcal{D}, f(-x) = - f(x)$$)，因此积分值为0，即 $$E(U) = 0$$。 

对于一般分布 $$X = \mu + \sigma U$$，由期望的线性性得到： $$E(X) = \mu + \sigma \times 0 = \mu$$

标准分布的方差$$ \text{Var}(U) = 1$$(证明略)，由方差的性质可得一般正态分布的方差 $$\text{Var}(X) = \text{Var}(\sigma U + \mu) = \sigma^2 \text{Var}(U) = \sigma^2$$
