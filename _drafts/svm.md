---
layout:     post
title:      Support Vector Machine
date:       2020-07-09
tags: [svm]
categories: 
- machine learning
---
TODO: Translate to CN. 

<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='25px'> What is SVM exactly ?

<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='25px'> 
* 一种多功能机器学习模型，可用于线性或线性**分类、回归及异常值检测**
* 对分类任务而言，SVM 将实例表示为空间上的点，寻找一条良好的决策边界间隔使得??? (不是任意的决策边界，而是能最大化...)
* 适合于复杂、中小规模数据集


---

<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='25px'> What is support vector ?
<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='25px'> 所谓向量，即是样本点 (每个样本都是一行数据，由向量表示)。支持向量即落在决策边界上的点 
(T2improved)

TODO: support vector determines decision boundary or the other way around ? 即先有决策边界还是先有支持向量?

---
<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='25px'> Ques: 可以多分类任务吗? 
<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='25px'> ...



<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='25px'> Pros and Cons ?

<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='25px'> Pros:
* xx

Cons:
* 对特征缩放敏感。因为 SVM 试图最大化类别之前的间隔，如果训练集没有进行特征缩放，那么 SVM 会倾向于忽略数据值较小的特征.


---

<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='25px'> Capable of multi-class classification ?

<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='25px'> Yes

---

<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='25px'> What is kernel trick ? How it works?

<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='25px'> 

---

<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='25px'> What is similarity function ? Difference with kernel trick ?

<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='25px'> 

---


# 线性可分 SVM (硬间隔模型)
对于分类任务，所有实例都被正确的分到决策边界的两边，这称为硬间隔分类。硬间隔分类存在两个问题: 
1. 只对线性可分问题越作用;
2. 对异常值敏感

## 最大间隔超平面
假设线性可分数据集合为 $$ T = \{ (x_i, y_i), i = 1,...,n\} $$，其中 $$x_i \in R^n, y_i \in \{-1, 1\}$$。考虑超平面 $$f(x): wx+b = 0$$，它将特征空间(正确的)分成两类，法向量 $$w$$ 指向的一侧为正类，另一侧为负类。再考虑函数 $$f(x)$$ 两侧的最近的两点 $$x^-, x^+$$ s.t. $$f(x^-) = -1, f(x^+) = 1$$ 且 $$ x^+ = x^- + r \cdot w $$。那么就有 
$$
\begin{aligned}
w \cdot (x^- + r \cdot w) + b & = 1\\
r ||w||^2 + w x^- + b & = 1\\
r ||w||^2  = 2 \\
r = \frac{2}{||w||^2}
\end{aligned}
$$
<img src="https://i.loli.net/2020/07/16/mn6ctfRvNlq7idY.png" width="350px">

$$r = \frac{2}{||w||^2}$$ 为超平面距离样本点的最小距离。 

SVM 的基本思想: **求解能够正确划分训练数据集并且几何间隔最大的分离超平面**。对于线性可分数据集来说，线性可分分离超平面有无穷多个，但几何间隔最大的只有一个。 
一个线性可分 SVM 算法即是要训练一个分类器，满足以下两点约束 
1. 所有数据被正确分类 
2. 间隔尽可能大 

这可由以下约束优化问题表示: 
$$
\begin{aligned}
    \underset{w}{\text{max }} r, \text{即 } \underset{w}{\text{min }} ||w||^2\\
    y_i (w x_i + b) - 1 \geq 0, i = 1, ..., N
\end{aligned}
$$

将这个问题作为原始最优化问题，通过拉格朗日对偶性，可通过求解其对偶问题来得到原始问题的最优解。 定义拉格朗日函数为 

$$
\begin{aligned}
    \mathcal{L}(w, b, \alpha) = \frac{1}{2} ||w||^2 - \sum\limits_{i=1}^{n} \alpha_i y_i (w x_i + b) + \sum\limits_{i=1}^{n} \alpha_i
\end{aligned}
$$

其中，$$ \alpha_i \geq 0, i=1, ..., n$$ 为拉格朗日乘子。原问题的对偶问题是极大极小问题 
$$
\begin{aligned}
    \underset{\alpha}{\text{max }} \underset{w, b}{\text{min }} \mathcal{L}(w, b, \alpha) 
\end{aligned}
$$

求解过程:
1. 求 $$  \underset{w, b}{\text{min }} \mathcal{L}(w, b, \alpha) $$
    * 分别对 $$ w, b $$ 求偏导 $$\nabla_w \mathcal{L} = w - \sum\limits_{i=1} \alpha_i y_i x_i = 0$$,  $$\nabla_b \mathcal{L} = - \sum\limits_{i=1} \alpha_i y_i = 0$$
    * 可得 $$ w = \sum\limits_{i=1} \alpha_i y_i x_i $$ (#2), $$ \sum\limits_{i=1} \alpha_i y_i = 0 $$ (#3)
    * 将公式 #2, #3 代入到拉格朗日函数中可得 
    $$
    \begin{aligned}
        \mathcal{L} & = \frac{1}{2} \sum\limits_i \sum\limits_j \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) - \sum\limits_i \alpha_i y_i ((\sum\limits_j \alpha_j y_j x_j )\cdot x_i + b) + \sum\limits_i \alpha_i\\
            & = - \frac{1}{2} \sum\limits_i \sum\limits_j \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) + \sum\limits_i \alpha_i
    \end{aligned}
    $$
2. 至此仅余下关于 $$\alpha$$ 的公式，问题 $$\underset{\alpha}{\text{max }} \mathcal{L} $$ 进而转化为 
    $$
    \begin{aligned}
            \underset{\alpha}{\text{min }}  \frac{1}{2} \sum\limits_i \sum\limits_j \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) - \sum\limits_i \alpha_i \\
            \text{s.t., } \sum\limits_i \alpha_i y_i = 0 \\ 
            \alpha_i \geq 0 
    \end{aligned}
    $$

> [定理] 设 $$\alpha^* = (\alpha_1^*, ..., \alpha_n^*)$$ 为以上问题的解，则存在 $$j\ge0$$，使得 $$ w^* = \sum\limits_i \alpha_i^* y_i x_i, b^* = y_j - \sum\limits_i \alpha_i^* y_i (x_i \cdot x_j) $$ 为原始最优化问题的解。 

由此，也可知 $$w, b$$ 仅由值大于 0 的$$\alpha_i$$ ($$\alpha_i$$要么大于 0 要么等于 0) 及其对应的样本点 $$(x_i, y_i)$$ 决定。 这些样本点也称为支持向量。


# 线性 SVM (软间隔模型)
现在中训练数据集往往是线性不可分的，意味着存在样本点 $$(x_i, y_i)$$ 不能满足函数间隔大于等于 1 的约束条件。为了解决这个问题，可能考虑引入松弛变量 $$\xi_i \ge 0$$，即约束为 $$y_i (w x_i + b) \geq 1 - \xi_i, i = 1, ..., n$$。

参照线性可分 SVM 的求解过程，线性 SVM 的求解问题转化求解以下凸二次规化问题:

$$
\begin{aligned}
    \underset{w}{\text{min }} \frac{1}{2} ||w||^2 + C \sum\limits_{i=1} \xi_i  \\
    y_i (w x_i + b) - 1 + \xi_i \geq 0, i = 1, ..., n \\
    \xi_i \geq 0, i = 1, ..., n
\end{aligned}
$$
其中目标函数中的 $$C\ge0$$ 为惩罚系数，值越大对误分类的惩罚也越大。最小化目标函数有两层含义: **1. 几何问题尽可能大; 2. 误分类样本点个数尽可能少**。前者要求 $$C$$ 尽可能小，而后者要求 $$C$$ 尽可能大。 也即是 $$C$$ 是调节二者的系数(一般手动选择)。 

以上优化问题可转化为:
$$
\begin{aligned}
            \underset{\alpha}{\text{min }}  \frac{1}{2} \sum\limits_i \sum\limits_j \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) - \sum\limits_i \alpha_i &(\#3)\\
            \text{s.t., } \sum\limits_i \alpha_i y_i = 0 \\ 
            0 \leq \alpha_i \leq C
    \end{aligned}
$$

求得其最优解 $$\alpha^* = (\alpha_1^*, ..., \alpha_n^*)$$，选择 $$\alpha^*$$ 的一个分量 $$\alpha_j^*, 0 \leq \alpha_j^* \leq C$$，取 $$ w^* = \sum\limits_i \alpha_i^* y_i x_i, b^* = y_j - \sum\limits_i \alpha_i^* y_i (x_i \cdot x_j) $$ ，则超平面 $$w^* x + b ^* = 0$$ 即为训练数据集在惩罚系数 $$C$$ 下的最大几何间隔分离超平面。 

## Hinge 损失函数

线性 SVM 的学习也可以通过求解以下问题得到:
$$
\begin{aligned}
    \underset{w, b}{\text{min }} \sum\limits_{i=1}[1 - y_i(w x_i + b)]_+ + \lambda ||w||^2 & (\#4)
\end{aligned}
$$
第一项是经验损失或者经验风险，函数 $$L(y(wx+b)) = [1 - y(wx+b)]_+$$ (即 $$\text{max}(0, 1 - y(wx+b))$$) 称为 Hinge 损失函数。直观解释为，当样本点被正确分类且函数间隔不小于 1 时，损失为 0，不然损失为 $$1 - y_i(wx_i+b)$$。

* 求解该问题与求解上节凸二次规化问题（#3）等价。 

证明: (#3)=>(#4)，对原目标函数乘系数 $$2\lambda$$ 不影响问题的解，令 $$C = \frac{1}{2} \lambda$$ ，则 (#3) 转为化以下问题
$$
\begin{aligned}
    \underset{w}{\text{min }} \lambda ||w||^2 + \sum\limits_{i=1} \xi_i  & (\#5) \\
    y_i (w x_i + b) - 1 + \xi_i \geq 0, i = 1, ..., n \\
    \xi_i \geq 0, i = 1, ..., n
\end{aligned}
$$
下的的两条约束可缩短为  $$\xi_i = \text{max}(0, 1 - y_i(w x_i + b))$$，即 $$\xi_i = [1 - y_i(w x_i + b)]_+$$，代入到 (#5) 即得 (#4)。

反之，(#4)=>(#3)，令 $$\xi_i = [1 - y_i(w x_i + b)]_+$$，则 $$\xi_i \geq 0 \wedge \xi_i \geq 1 - y_i(w x_i + b)$$。且对任意常数 $$C$$, (#4) 等价于 $$ \underset{w, b}{\text{min }} C \sum\limits_{i=1} \xi_i + C \cdot \lambda ||w||^2  $$，令 $$ \lambda = \frac{1}{2C} $$，则 (#3) 成立。

软间隔模型在尽可能宽的决策边界与尽可能少的错误分类之间寻求一个平衡.



# 非线性 SVM
对于 $$R^n$$ 上线性不可分数据集合，如果存在 $$R^n$$ 上的超平面可将其分开，则称其为非线性可分问题。 
求解非线性的问题的一个方法是，对数据集进行非线性变换，将非线性问题转化为线性问题。比如二维平面上通过曲面可分的样本集，可通过曲面方程映射将样本数据映射到线性可分的新数据集上。 

核技巧的思想，通过非线性变换将输入空间映射到一个特征空间(希尔伯特空间) $$\mathcal{H}$$，使得输入空间上的超曲面模型对应到特征空间 $$\mathcal{H}$$ 上的超平面模型。



## 核技巧
向数据表示中添加非线性特征，可以让线性模型变得更强大。但通常来说，我们并不知道需要添加哪些特征，而且添加很多特征的计算开销可能会很大。而核技巧(kernel trick)可以让我们在更高维空间中学习分类器，原理是**直接计算扩展特征表示中数据点之间的距离**(内积)，而不用实际对扩展进行计算。

## 核函数 
常用核函数 
1. 多项式核函数， $$ K(x,z) = (x \cdot z + 1) ^ p$$
2. 高斯核函数(Guassian Kernel Funcction) $$ K(x,z) = \exp(- \frac{||x - z||^3}{2 \sigma^2}) $$，对应的 SVM 称为高斯径向基函数(radial basis function, RBF)分类器


## 计算复杂性
以 scikit-learn 框架而言
* `LinearSVC` 类基于 `liblinear` 库，它实现了线性 SVM 的优化算法。尽管它不支持核技巧，但计算复杂度与样本空间大小及特征数量呈线性，即 $$O(m \cdot n)$$
* `SVC` 类基于 `libsvm` 库，它实现了支持核技巧的算法，复杂度通常介于 $$O(m^2  \cdot n)$$ 与 $$O(m^3 \cdot n)$$ 之间，其中 $$m, n$$ 分别为样本个数及特征个数

## 小结

* 优点:
    * 非常强大，在各种数据集上的表现都很好
* 缺点:
    * 对样本个数的缩放表现不好 ??
    * 预处理数据和调参都需要非常小心
    * 解出的模型(参数)很难解释 

