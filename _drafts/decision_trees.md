---
layout:     post
title:      Decision Trees
date:       2019-11-25
tags: [decision tree, supervised learning]
categories: 
- machine learning
---

# Decision Trees 
* 计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可能处理不相关特征数据 
* 可能会过拟合

决策树的学习通常分为三部分:特征选择,决策树的生成和决策树的剪枝.

## ID3
Iternative Dichotomizer 3, 由 Ross Quinlan (Quinlan, J. R. 1986.  Induction of Decision Trees. Mach. Learn. 1, 1 (Mar. 1986), 81-106.) 在1986年提出。

ID3 决策树可以有多个分支，但是不能处理特征值为连续的情况。决策树是一种贪心算法，每次选取的分割数据的特征都是当前的最佳选择，并不关心是否达到全局最优。在 ID3 中，每次根据**最大信息熵增益**选取当前最佳的特征来分割数据，并按照该特征的所有取值来切分，也就是说如果一个特征有3种取值，数据将被切分3份，一旦按某特征切分后，该特征在之后的算法执行中，将不再起作用。

划分数据集的大原则是：**将无序的数据变得更加有序**。ID3使用**信息增益**(数据集划分前后信息发生的变化)的方法来划分。
要计算信息增益，我们需要一种度量集合信息的方式，比如香农熵(简称熵)。熵定义为信息的期望值，对于待分类的事物，符号$$x_i$$的信息定义为
$$l(x_i) = - \text{log}_2p(x_i)$$，其中$$p(x_i)$$为该分类的的概率。由这些分类构成的集合的熵 $$H = -\Sigma_{i=1}^n p(x_i) \text{log}_2 p(x_i) $$ 。 

从物理意义上直观的讲，熵对应一个系统的混乱与不一致程度，熵越大，表明这个系统越混乱。
信息增益刻画的是：熵的减少或者数据无序度的减少。 

ID3的一般思路是：
1. 测量集合数据的熵 

    $$H(D) = -\sum\limits_{k=1}^K \frac{|C_k|}{|D|} \text{log}_2 \frac{|C_k|}{|D|}$$
2. 寻找最优方案(特征)划分数据集，即计算每个特征 $$f_i$$ 对数据集 $$D$$ 的经验条件熵 

    $$H(D|f_i) = \sum\limits_{j=1}^n \frac{|D_j|}{|D|} H(D_j) $$

    然后使用可得最大信息增益的特征 $$f_i$$ 对数据集进行划分 

    $$f_\text{best} = \underset{f_i}{\arg \max} \ g(D, f_i) = \underset{f_1}{\arg \max} \ (g(D) - g(D|f_i))$$
3. 对子集进行递归划分直到子集中所有数据属于同一个分类，或者特征耗尽


### Gini Index or Information Gain
信息增益通过求和**分类概率与概率的对数(底为2)的乘积**来计算，此方式倾向于选择属性值较多的特征。 
Gini 系数通过减去每个类别概率平方的和来计算，it favors larger partitions (????)。 
Gini 系数代表了模型的不纯度，基尼系数越小，则不纯度越低，特征越好。 具体的，假设有 $$K$$ 个类别，第 $$k$$ 个类别的概率为 $$p_k$$， 则基尼系数的表达式为：

$$\text{Gini}(p) = \sum\limits_{k=1}^K p_k (1-p_k) = 1 - \sum\limits_{k=1}^K p_k^2 $$

那么对于样本 $$D$$，有 $$K$$ 个类别，第 $$k$$ 个类别的数量为 $$C_k$$， 则样本 $$D$$ 的基尼系数的表达式为：

$$\text{Gini}(D) = 1 - \sum\limits_{k=1}^K (\frac{|C_k|}{|D|})^2 $$

特别的，对于样本 $$D$$，如果根据特征 $$f$$ 的某个值，把 $$D$$ 分成 $$D_1$$ 和 $$D_2$$ 两部分，则在特征 $$f$$ 的条件下，$$D$$ 的基尼系数表达式为：

$$\text{Gini}(D, f) = \frac{|D_1|}{|D|} \text{Gini}(D_1) + \frac{|D_2|}{|D|} \text{Gini}(D_2) $$

[TODO](https://www.cnblogs.com/pinard/p/6053344.html)

### ID3 Python Implementation 

[Deicision-Tree-ID3-Python3]({{site.baseurl}}/codes/decision_tree.py.txt)

ID3 的缺陷：
1. 数据集不够大时，很容易过拟合
2. 每次只能考察一个特征来作决策
3. 无数处理(连续)的数值特征及缺失值

## C4.5
C4.5 由 Ross Quinlan 于1993年提出对 ID3 算法的扩展。ID3 采用的信息增益度量存在一个内在偏置，它优先选择有较多属性值的特征，因为属性值多的特征可能会有相对较大的信息增益 (信息增益反映的给定一个条件以后不确定性减少的程度，数据集分得越细，确定性越高。相对的，条件熵越小，信息增益越大)。 避免这个不足的一个度量就是不用信息增益来选择 feature，而是用信息增益比率(gain ratio)，增益比率通过引入一个被称作分裂信息(split information)的项来惩罚取值较多的 feature， 分裂信息用来衡量 feature 分裂数据的广度和均匀性:

$$\text{SI}(D, f) =  -\sum\limits_{i=1}^n \frac{|D_i|}{|D|} \text{log}_2 \frac{|D_i|}{|D|} $$

$$ \text{GR}(D, f) = \frac{g(D, f)}{\text{SI}(D, f)} $$

where SI stands for Split Information, and GR for Gain Ratio.

### ID3 VS. C4.5
C4.5 是 ID3 算法的扩展。
 
<!-- 1. ID3 uses information gain whereas C4.5 uses gain ratio for splitting.  -->
1. ID3 使用信息增益，而 C4.5 使用增益比率
2. ID3 每次划分分组时都会消耗特征，即划分数据分组之后特征数目会减少，而C4.5 & CART并不总是消耗特征
3. TODO

C4.5 较 ID3 的优势
<!-- 1. accepts both continuous and discrete features -->
1. 可以处理连续属性值
<!-- 2. handles incomplete data points;  -->
2. 可以处理缺失值 
    - 丢弃存在缺失值的样本
    - 补上该属性的均值或者频率最高的值
<!-- 3. solves over-fitting problem by (very clever) bottom-up technique usually known as "pruning";  -->
3. 通过预剪枝来解决过拟合问题
<!-- 4. different weights can be applied the features that comprise the training data. -->


# CART 
CART, Classification And Regression Trees, 分类决策树。

优点，可以对复杂和非线性的数据建模；缺点是，结果不易理解。

CART 采用二元切分来处理连续型变量，即每次把数据集切成两份，如果数据的某特征值大于切分所要求的值，那么这些数据就进入树的左子树，反之进入树的右子树。

如何度量连续型数值的不一致度？首先计算所有数据的均值，然后计算每条数据的值到均值的差值(绝对值或者平方值)。 

采用 Gini 系统来划分属性。


# 小结

| 算法   | 支持模型 | 特征选择方法 | 连续值处理 | 缺失值处理 | 剪枝 | 树结构 |
| :---- | :-----  | :--------- | :------- | :-- | :-- | :-- |
| ID3  | 分类  | 信息增益  | No | No | No | 多叉树 |
| C4.5 | 分类  | 信息增益比	| Yes| Yes| Yes | 多叉树 |
| CART	| 分类，回归 |	基尼系数，均方差 | Yes | Yes | Yes | 二叉树 |

算法不足之处：

1. 无论是 ID3, C4.5 还是 CART，在做特征选择的时候都是选择最优的一个特征来做分类决策，但有时候分类决策由一组特征来决定，得到的决策树会更加准确。这类决策树叫做**多变量决策树(multi-variate decision tree)**。在选择最优特征的时候，多变量决策树不是选择某一个最优特征，而是选择最优的一个特征线性组合来做决策。代表算法 OC1。