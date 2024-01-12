---
layout: post
title: Probability Distributions
date: 2023-07-12
tags: math distribution
categories: math
author: GaoangLiu
---
* content
{:toc}


# 幂律分布 
幂律分布（power law distribution）是分布密度函数为幂函数的分布，其概率密度函数满足：




$$
p(x) = C x^{-\alpha}
$$

其中，$$C$$ 为归一化系数，$$\alpha$$ 为幂指数，$$x$$ 为随机变量。根据密度函数可以得知，$$x$$ 取值为 $$m$$ 的概率是 $$x$$ 取值为 $$n$$ 的概率的 $$(\frac{m}{n})^-\alpha$$ 倍。$$x$$ 取值越大，概率越小，但是概率的下降速度是随着 $$x$$ 的增大而减小的。幂律分布的特点是长尾分布，即随机变量的取值在某个值之后，概率密度函数以幂函数的形式递减。幂律分布的期望和方差存在的条件是 $$\alpha > 2$$ 和 $$\alpha > 3$$，分别为：


对于定义域 $$x \geq x_\text{min}$$，分布的期望:

$$\begin{aligned}
E(X) &= \int_{x_\text{min}}^\infty x p(x) dx \\\
&= \int_{x_\text{min}}^\infty C x^{1-\alpha} dx \\\
&= \frac{C}{2-\alpha} x^{2-\alpha} \bigg|_{x_\text{min}}^\infty \\\
&= \frac{C}{\alpha-2} x_\text{min}^{2-\alpha}, \text{ when } \alpha > 2
\end{aligned}$$

对应的幂律分布的方差：

$$\begin{aligned}
\text{Var}(X) &= E(X^2) - E(X)^2 \\\
&= \int_{x_\text{min}}^\infty x^2 p(x) dx - \left(\frac{C}{\alpha-2} x_\text{min}^{2-\alpha}\right)^2 \\\
&= \frac{C}{3-\alpha} x^{3-\alpha} \bigg|_{x_\text{min}}^\infty - \left(\frac{C}{\alpha-2} x_\text{min}^{2-\alpha}\right)^2 \\\
&= \frac{C}{\alpha-3} x_\text{min}^{3-\alpha} - \left(\frac{C}{\alpha-2} x_\text{min}^{2-\alpha}\right)^2, \text{ when } \alpha > 3
\end{aligned}$$


# 指数分布 

指数分布（exponential distribution）是一种连续概率分布，其概率密度函数为：

$$\begin{aligned}
f(x) &= \lambda e^{-\lambda x}, x \geq 0 \\\
&= 0, x < 0
\end{aligned}$$

其中 $$\lambda > 0$$ 为分布的参数，称为速率参数(rate parameter)。指数分布的随机亦是只可能取非负数，所有常被用作各种“寿命”的分布，比如设备寿命、客户到达间隔、放射性衰变时间等等。指数分布的期望和方差为：

$$\begin{aligned}
E(X) &= \frac{1}{\lambda} \\\
\text{Var}(X) &= \frac{1}{\lambda^2}
\end{aligned}$$

指数分布的一个典型的特征是无记忆性(memoryless)，即对于任意 $$s, t > 0$$，有：

$$
P(X > s + t | X > s) = P(X > t)
$$

这个特性的意思是，如果一个随机变量服从指数分布，那么它的概率密度函数在 $$s$$ 时刻的取值与 $$s+t$$ 时刻的取值无关，即 $$s$$ 时刻的取值不会影响 $$s+t$$ 时刻的取值。

这个特性在实际中有很多应用，假设一台电子设备，它的寿命服从指数分布 $$X \sim Exp(\lambda)$$，已知它已经工作了 $$s$$ 个小时，那么它继续正常工作 $$t$$ 个小时的概率与它刚开始工作 $$t$$ 个小时的概率是一样的。看起来比较奇怪，甚至有点反直觉，但从概率推断一下就可以理解了。设 $$X$$ 为设备的寿命，$$s$$ 为已经工作的时间，$$t$$ 为继续工作的时间，那么，$$P(X \ge s)=1-P(X \le s)=1-F(s)=e^{-\lambda s}$$。

由于 $$\{X \ge t + s\} \subseteq \{X \ge s\}$$，因此：

$$\begin{aligned}
P(X > s + t | X > s) &= \frac{P(X > s + t, X > s)}{P(X > s)} \\\
&= \frac{P(X > s + t)}{P(X > s)} \\\
&= \frac{e^{-\lambda (s+t)}}{e^{-\lambda s}} \\\
&= e^{-\lambda t} \\\
&= P(X > t)
\end{aligned}$$

但动物的寿命，特别是人的寿命分布比较复杂，受到医疗、生活环境等多种因素的影响，不能简单的套用指数分布。