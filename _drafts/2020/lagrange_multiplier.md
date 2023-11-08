---
layout:     post
title:      Lagrange Multiplier
date:       2020-07-09
tags: [lagrange, svm]
categories: 
- Math
---

拉格朗日乘子法是一种寻找多元函数在其变量受到一个或多个条件的约束时的极值的方法。这种方法可以将一个有$$n$$个变量与$$k$$个约束条件的最优化问题转换为一个解有$$n + k$$个变量的方程组的解的问题。这种方法中引入了一个或一组新的未知数，即**拉格朗日乘数**，又称拉格朗日乘子、拉氏乘子。拉格朗日乘子作为转换后的方程，即约束方程中约束条件的系数。 

比如，在求 $$f(x,y)$$ 在 $$ g(x, y) = c $$ 时的最大值时，我们可以引入拉格朗日乘数 $$\lambda$$，这时我们只需要计算下列拉格朗日函数的极值：
$$
\begin{aligned}
    \mathcal{L}(x, y, \lambda) = f(x, y) + \lambda(g(x, y) - c)
\end{aligned}
$$

更一般的，对于 $$n$$ 元函数 $$ f(x_1, ..., x_n) $$ 及 $$k$$ 个约束 $$g_i(x_1, ..., x_n) = c_i$$，有：
$$
\begin{aligned}
    \mathcal{L}(x_1, x_2, ..., x_n, \lambda_1, ..., \lambda_k) = f(x_1, ..., x_n) + \sum_{i}^k \lambda_i(g(x_1, ..., x_n) - c_i)
\end{aligned}
$$

拉格朗日乘数法所得的极点会包含原问题的所有极值点，但并不保证每个极值点都是原问题的极值点。


---

## 例子
考察 $$f(x, y) = x^2 y $$ 在约束 $$x^2 + y^2 = 1$$ 下的极小值。

1. 只有一个约束，因此只需要添加一个拉格朗日乘子 $$\lambda$$. 构建拉格朗日函数 $$\mathcal{L}(x, y, \lambda) = x^2 y + \lambda(x^2 + y^2 - 1)$$
2. 求方程对各个变量的偏微分，得到一个方程组: 
$$
\begin{aligned}
    x^2+y^2=1\\
    2xy + 2\lambda x = 0\\
    x^2 + 2\lambda y = 0
\end{aligned}
$$
3. 最小值即为以上方程组的一个解 $$x = \sqrt{\frac{2}{3}}, y = -\sqrt{\frac{1}{3}}$$ 对应的值 $$ f(x, y) = - \frac{2}{3 \sqrt{3}}$$ 


## 拉格朗日对偶性
假设 $$f(x), c_i(x), h_j(x)$$ 为连续可微函数。考虑最优化问题 
$$
\begin{aligned}
    \underset{x \in R^n}{\text{min }} f(x)\\
    \text{s.t. } c_i(x) \leq 0, i \in [1, k] \\
    h_j(x) = 0, j \in [1, l]
\end{aligned}
$$

问题可以转化为广义拉格朗日函数 
$$\mathcal{L}(x, \alpha, \beta) = f(x) + \sum\limits_{i=1}^{k} \alpha_i c_i(x) + \sum\limits_{j=1}^{l} \beta_j h_j(x) $$,
其中 $$\alpha_i, \beta_j$$ 称拉格朗日乘子，$$\alpha_i \geq 0$$. 考虑 $$x$$ 的函数 
$$
\begin{aligned}
    \theta_P(x) = \underset{\alpha, \beta: \alpha_i \geq 0}{\text{max }} \mathcal{L}(x, \alpha, \beta)
\end{aligned}
$$

P: primary 表示原始问题。 如果 $$x$$ 不满足以上问题的任一条件，总有 $$ \theta_P(x) = \infty$$, 反之有 $$\theta_P(x) = f(x)$$. 

考虑极小值问题 
$$
\begin{aligned}
    \underset{x}{\text{min }} \theta_P(x) = \underset{x}{\text{min }} \underset{\alpha, \beta: \alpha_i \geq 0}{\text{max }} \mathcal{L}(x, \alpha, \beta)
\end{aligned}
$$
这个问题称为拉格朗日函数的极小极大问题，问题的解与原问题的解一致。记 $$p^* = \underset{x}{\text{min }} \theta_P(x)$$ 为原问题的最优解。

### 对偶问题
定义 
$$
\begin{aligned}
    \theta_D(\alpha, \beta) = \underset{x}{\text{min }} \mathcal{L}(\alpha, \beta, x)
\end{aligned}
$$
考虑其极大化问题，即:
$$
\begin{aligned}
    \underset{\alpha, \beta: \alpha_i \geq 0}{\text{max }}\theta_D(\alpha, \beta) = \underset{\alpha, \beta: \alpha_i \geq 0}{\text{max }}\underset{x}{\text{min }} \mathcal{L}(\alpha, \beta, x)
\end{aligned}
$$
问题称为拉格朗日极大极小问题。其对应的约束问题为: 

$$
\begin{aligned}
    \underset{\alpha, \beta: \alpha_i \geq 0}{\text{max }}\theta_D(\alpha, \beta) = \underset{\alpha, \beta}{\text{max }}\underset{x}{\text{min }} \mathcal{L}(x, \alpha, \beta) \\
    \text{s.t., } \alpha_i \geq 0, i=1,2,...,k
\end{aligned}
$$

此问题称为原问题的对偶问题，记其最优解为 $$d^*$$ 

### 原问题与对偶问题关第 
以下定理表明对偶问题的最优值不大于原问题的最优值(弱对偶性)。 

> 若两问题都存在最优解，那么 $$d^* = \underset{\alpha, \beta}{\text{max }}\underset{x}{\text{min }} \mathcal{L}(x, \alpha, \beta) \leq \underset{x}{\text{min }} \underset{\alpha, \beta: \alpha_i \geq 0}{\text{max }} \mathcal{L}(x, \alpha, \beta) = p^*$$ 。

推论:
* 假设 $$x^*, \alpha^*, \beta^*$$ 为对问题及对偶问题的可行解，且 $$p^* = d^*$$ (强对偶性)，那么 $$x^*, \alpha^*, \beta^*$$ 为原问题及对偶问题的最优解

考虑对偶问题的一好处是：当原始问题不好求解而对偶问题相对好求解的时候，这时我们就可以用求解对偶问题替代求解原始问题。并且更重要的是，对偶问题是一个凸优化问题，它的极值是唯一的(因为$$d^* \leq p^*$$)。这样无论一个问题是不是凸优化的问题，我们都能将其转化成凸优化的问题。

那么什么情况下有 $$ d^* = p^* $$? 其中一个为 KKT 条件(Karush-Kuhn-Tucker Conditions).

> 假设函数 $$f(x)$$ 和 $$c_i(x)$$ 都是凸函数，$$h_j(x)$$ 是仿射函数(由一阶多项式构成的函数)，并且不等式约束 $$c_i(x)$$ 是**严格可行**的，即存在 $$x$$，对所有的 $$i$$，都使得$$c_i(x)<0$$ (注意这里是严格要求小于0,而不是小于等于0)，则存在 $$x^*, \alpha^*,\beta^*$$ 使得 $$x^*$$ 是原始问题的解，$$α^*,β^*$$ 是对偶问题的解，并且 $$d^* = p^* = \mathcal{L}(x^*, \alpha^*, \beta^*)$$.

以上定理给出最优解存在性的充分条件，KKT 条件给出了解存在的充分必要条件:

> 假设函数 $$f(x)$$ 和 $$c_i(x)$$ 都是凸函数，$$h_j(x)$$ 是仿射函数(由一阶多项式构成的函数)，并且不等式约束 $$c_i(x)$$ 是**严格可行**的，即存在 $$x$$，对所有的 $$i$$，都使得$$ c_i(x)<0 $$，则 $$x^*, \alpha^*,\beta^*$$ 分别是原始问题及对偶问题的解的充要条件是以下 KKT 条件:
1. $$ \nabla_x \mathcal{L}(x^*, \alpha^*, \beta^*) = 0 $$
2. $$ \alpha_i^* c_i (x^*) = 0 $$
3. $$ c_i (x^*) \leq 0 $$
4. $$ \alpha_i \geq 0 $$
5. $$ h_j(x^*) = 0 $$

---

[QUES] 什么是凸优化问题? 为什么对偶问题是一个凸优化问题？


# 凸优化问题
## 凸集
对于 $$n$$ 维空间中点的集合 $$C$$，如果对集合中的任意两点$$x, y$$，以及实数 $$0 \leq \theta \leq 1$$，都有 
$$
\begin{aligned}
    \theta x + (1 - \theta) y \in C
\end{aligned}
$$
即任意两点的凸组合仍在集合内，则称该集合为凸集。

有意义的结论:
1. 如果一个优化问题是不带约束的优化，则其优化变量的可行域是一个凸集。
2. 如果一组约束是线性等式约束，则它确定的可行域是一个凸集。
3. 如果一组约束是线性不等式约束，则它定义的可行域是凸集。
4. 多个凸集的交集还是凸集(并不成立)

而我们遇到的优化问题中，可能有多个等式和不等式约束，只要每个约束条件定义的可行域是凸集，则同时满足这下约束条件的可行域还是凸集。

## 凸函数 
在函数 $$f$$ 的定义域内，如果对任意的 $$x, y$$，以及实数 $$0 \leq \theta \leq 1$$，都有 
$$
\begin{aligned}
    f(\theta x + (1 - \theta) y) \leq \theta f(x) + (1 - \theta) f(y)
\end{aligned}
$$
则函数称为凸函数。如果公式中换成严格小于，则称严格凸函数。

凸函数判定规则
* 一阶判定规则，$$ f(y) \geq f(x) + \nabla f(x)^T (y-x) $$，对应几何解释，函数在任意点的切线都在函数下方
* 二阶判定规则
    * 一元函数，$$f^{''}(x) \geq 0$$，即二阶导数大于等于 0。如果严格大于 0，则函数是严格凸的。
    * 对于多元函数，如果它是凸函数，则其 Hessian 矩阵为半正定矩阵。如果 Hessian 矩阵是正定的，则函数是严格凸函数

### Hessian 矩阵
Hessian 矩阵是由多元函数的二阶偏导数组成的矩阵，是一个对称矩阵。如果函数 $$f(x_1, ..., x_n)$$ 二阶可导，则 Hessian 矩阵定义为:
$$
\begin{equation*}
H =  
\begin{pmatrix}
\frac{\partial^2 f}{\partial x_1 ^2} &  \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} &  \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots  & \vdots  & \ddots & \vdots  \\
\frac{\partial^2 f}{\partial x_n \partial x_1} &  \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} 
\end{pmatrix}
\end{equation*}
$$

根据多元函数极值判别法，假设多元函数在点 $$X$$ 的梯度为0，即 $$X$$ 是函数的驻点，则有：
1. 如果 Hessian 矩阵正定，函数在该点有极小值
2. 如果 Hessian 矩阵负定，函数在该点有极大值
3. 如果 Hessian 矩阵不定，则不是极值点（鞍点）

正定的定义为，如果对任意非 0 的 n 维向量$$x^T$$，
* 都有 $$x^T A x \ge 0$$ (严格大于)，则称矩阵 $$A$$ 为正定矩阵(类似于二阶导数大于 0)
* 都有 $$x^T A x \geq 0$$，则称矩阵 $$A$$ 为半正定矩阵
* 都有 $$x^T A x \le 0$$，则称矩阵 $$A$$ 为负定矩阵

### 下水平集
给定一个凸函数 $$f$$ 以及一个实数 $$\alpha$$ ，函数的 $$\alpha$$ 下水平集（sub-level set）定义为函数值小于等于 $$\alpha$$ 的点构成的集合：

$$
\begin{aligned}
    \{ x \in D(f): f(x) \leq \alpha \}
\end{aligned}
$$

根据凸函数的定义，很容易证明该集合是一个凸集。藉此概念我们可以确保优化问题中由凸函数构成的不等式约束条件定义的可行域仍然是凸集。


## 凸优化问题
凸优化问题即是研究定义在凸集上的凸函数的最小化问题， 一般写作
$$
\begin{aligned}
    \underset{x \in C}{\text{min }}f(x)
\end{aligned}
$$
其中，$$x, C, f$$ 分别表示优化变量、优化变量的可行域及凸目标函数。另一种通用写法: 

$$
\begin{aligned}
    {\text{min }}f(x) \\
    c_i(x) \leq 0, i = 1, ..., m\\
    h_j(x) = 0, j = 1, ...,p
\end{aligned}
$$

其中 $$c_i(x) $$ 是不等式约束函数，为凸函数，$$h_j(x)$$ 是等式约束函数，为仿射函数。

凸优化问题有一个重要的特性：局部最优解即是全局最优解。求解方法包括 GD、牛顿法、拟牛顿法等。