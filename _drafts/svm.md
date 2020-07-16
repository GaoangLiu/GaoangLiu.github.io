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
考虑超平面 $$f(x): wx+b = 0$$，它将特征空间分成两类，一类是正类，一类是负类。法向量 $$w$$ 指向的一侧为正类。再考虑函数 $$f(x)$$ 两侧的最近的两点 $$x^-, x^+$$ s.t. $$f(x^-) = -1, f(x^+) = 1$$ 且 $$ x^+ = x^- + r \cdot w $$。那么就有 
$$
\begin{aligned}
w \cdot (x^- + r \cdot w) + b & = 1\\
r ||w||^2 + w x^- + b & = 1\\
r ||w||^2  = 2 \\
r = \frac{2}{||w||^2}
\end{aligned}
$$
<img src="https://i.loli.net/2020/07/16/mn6ctfRvNlq7idY.png" width="350px">

最大间隔分类器需要满足以下两点约束 
1. 所有数据被正确分类 
2. 间隔尽可能大
这可由以下约束优化问题表示: 
$$
\begin{aligned}
    \subset{w}{\text{min }} ||w||^2\\
    y_i (w x_i + b) - 1 \geq 0, i = 1, ..., N
\end{aligned}
$$



# 线性 SVM (软间隔模型)
软间隔模型在尽可能宽的决策边界与尽可能少的错误分类之间寻求一个平衡.

# 非线性 SVM
... 
## 核技巧
向数据表示中添加非线性特征，可以让线性模型变得更强大。但通常来说，我们并不知道需要添加哪些特征，而且添加很多特征的计算开销可能会很大。而核技巧(kernel trick)可以让我们在更高维空间中学习分类器，原理是**直接计算扩展特征表示中数据点之间的距离**(内积)，而不用实际对扩展进行计算。


## 损失函数 
Hinge 损失函数 

。。 

$$\text{argmax}_{w, b} $$

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

