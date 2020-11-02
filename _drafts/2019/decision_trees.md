---
layout:     post
title:      Decision Trees
date:       2019-11-25
tags: [decision tree, supervised learning]
categories: 
- machine learning
---


## 决策树的学习
本质上从训练数据集中归纳出一组分类规则，策略是以损失函数为目标函数的最小化。 当损失函数确定以后，学习问题转化为在损失函数意义下选取最优决策树的问题。这个问题是 NP 完全问题，现实中的方法一般采用启发式算法，近似求解这一优化问题，得到的决策树是次最优的。 

决策树学习 3 部曲: **特征选择**、**决策树的生成**、**决策树的剪枝**

特征选择的准则一般是：信息增益或者信息增益比。如果利用一个特征进行分类的结果与随机分类的结果没有多大区别，则这个特征对分类没有多大意义 。

## 树的生成策略
### 信息增益
给定随机变量 $$X$$，熵(entropy) 是随机变量不确定性的度量。
设 $$X$$ 取值于 $$\{x_1, ..., x_n \}$$，在值 $$x_i$$ 的概率为 $$p(x_i)$$，则 $$X = x_i$$的信息定义为: $$l(x_i) = - \text{log}_2p(x_i)$$

熵则定义为信息的期望值，即
$$
\begin{aligned}
    H = -\sum\limits_{i=1}^n p(x_i) \log p(x_i) 
\end{aligned}
$$ 

从物理意义上直观的讲，熵对应一个系统的混乱与不一致程度，熵越大，表明这个系统越混乱。

对 DT 分类算法应用而言，信息增益刻画的是，在得知特征 $$X$$ 的情况下类别 $$Y$$ 的不确定性减少的程度。 形式化定义
> 特征 $$f$$ 对数据集 $$D$$ 的信息增益 $$g(D, f)$$，定义为集合 $$D$$ 的经验熵与给定条件 $$f$$ 的情况下 $$D$$ 的条件熵 $$H(D \vert f)$$ 之差，即 $$g(D, f) = H(D) - H(D \vert f)$$


### 信息增益比
由于信息增益反映的给定一个条件以后不确定性减少的程度，数据集分得越细，确定性越高。相对的，条件熵也越小，信息增益也越大。因此信息增益倾向于选择特征值较多的特征，避免这个不足的一个度量就是不用信息增益来选择特征，而是用信息增益比率(gain ratio)，增益比率通过引入一个被称作分裂信息(split information)的项来惩罚取值较多的特征， 分裂信息用来衡量特征分裂数据的**广度和均匀性**:

$$ \text{SI}(D, f) =  -\sum\limits_{i=1}^n \frac{|D_i|}{|D|} \text{log}_2 \frac{|D_i|}{|D|} $$

$$ \text{GR}(D, f) = \frac{g(D, f)}{\text{SI}(D, f)} $$

SI 表示分裂信息。


# Cons & Pros
## Pros
* 算法简单易于理解，模型(树)可视化，输出结果易于理解
* 几乎不需要对数据进行预处理，不需要标准化数据、移除缺失值
* 训练好的模型预测新的数据速度很快 (对数时间)
* 可以处理数值型及分类型数据

## Cons
* 易过拟合。可通过后剪枝、设置最大深度、最小叶结点大小等方式来避免这个问题
* 树的结构对数据变动敏感。数据集的微小变动可能导致生成完全不同的树结构 (可通过 ensemble 多棵树来缓解这个问题)
* 基于熵或者 Gini coefficient 分割标准来划分数据集在每一步只得到一个局部最优解，全局最优解是 NPC 问题 (可通过 bootstrap + ensemble 来提高模型性能)
* 数据不平衡情况下，容易生成 biased tree
* 难以学习 `XOR`, multiplexer 等概念及问题


---

# ID3 Algorithm

Iternative Dichotomizer 3, 由 Ross Quinlan (Quinlan, J. R. 1986.  Induction of Decision Trees. Mach. Learn. 1, 1 (Mar. 1986), 81-106.) 在1986年提出。

ID3 决策树可以有多个分支，但是不能处理特征值为连续的情况。

决策树是一种贪心算法，每次选取的分割数据的特征都是当前的最佳选择，并不关心是否达到全局最优。在 ID3 中，每次根据**最大信息熵增益**选取当前最佳的特征来分割数据，并按照该特征的所有取值来切分，也即是说如果一个特征有 `N` 种取值，数据将被切分 `N` 份，一旦按某特征切分后，该特征在之后的算法执行中，将不再起作用。

划分数据集的大原则是：**将无序的数据变得更加有序**。ID3使用**信息增益**(数据集划分前后信息发生的变化)的方法来划分。

ID3的思路是：
1. 测量集合数据的熵

    $$H(D) = -\sum\limits_{k=1}^K \frac{|C_k|}{|D|} \text{log}_2 \frac{|C_k|}{|D|}$$
2. 寻找最优方案(特征)划分数据集，即计算每个特征 $$f_i$$ 对数据集 $$D$$ 的经验条件熵 

    $$H(D|f_i) = \sum\limits_{j=1}^n \frac{|D_j|}{|D|} H(D_j) $$

    然后使用可得最大信息增益的特征 $$f_i$$ 对数据集进行划分 

    $$f_\text{best} = \underset{f_i}{\arg \max} \ g(D, f_i) = \underset{f_1}{\arg \max} \ (g(D) - g(D|f_i))$$
3. 对子集进行递归划分直到子集中所有数据属于同一个分类，或者特征耗尽


### Gini Index or Information Gain ?
1. Generally, the performance will not change whether you use Gini impurity or Entropy ([*1])
2. However, entropy might be a little **slower to compute** (because it makes use of the logarithm).

[*1] An article [**Theoretical comparison between the gini index and information gain criteria**](https://bit.ly/2ACcaeP) claims that, "... We found that they disagree only in 2% of all cases, which explains why most previously published empirical results concluded that it is not possible to decide which one of the two tests performs better." 

* Entropy:
    * $$H(E) = - \sum_{j=1}^c p_j \log p_j$$, 信息增益通过求和**分类概率与概率的对数(底为2)的乘积**来计算，此方式倾向于选择属性值较多的特征。 

* Gini:
    * $$Gini(E) = 1 - \sum\limits_{j=1}^c p_j^2 $$, Gini 系数通过减去每个类别概率平方的和来计算。 
    * Gini 系数代表了模型的**不纯度**，基尼系数越小，则不纯度越低，特征越好。 
    

Experiments:
1. We've conducted a simple experiment on calculating the `accuracy_score` with Decision Tree classifier on 100 randomly generated data sets. For each data set, we build a classifier with `gini` and `entropy` as the split criterion, with all other parameters be the same. Our result (see the following image) confirms the above claim that NONE of two criteria outperform the other.

2. Source code (`ipynb` file) of the experiment can be found [here](https://gaoangliu.github.io/codes/mlmodels/Gini_or_Entropy.ipynb)


<img src="https://i.loli.net/2020/06/15/VUvf5Bqj9GHspAQ.png" width="700px" alt='gini vs entropy'>

### ID3 Python Implementation 

[Deicision-Tree-ID3-Python3](https://gaoangliu.github.io/codes/mlmodels/decision_tree.ipynb)


ID3 的缺陷：
1. 数据集不够大时，很容易过拟合
2. 每次只能考察一个特征来作决策
3. 无数处理(连续的)数值特征及缺失值

---

# C4.5
C4.5 由 Ross Quinlan 于1993年提出对 ID3 算法的扩展。ID3 采用的信息增益度量存在一个内在偏置，它优先选择有较多属性值的特征，因为属性值多的特征可能会有相对较大的信息增益 (信息增益反映的给定一个条件以后不确定性减少的程度，数据集分得越细，确定性越高。相对的，条件熵越小，信息增益越大)。 避免这个不足的一个度量就是不用信息增益来选择 feature，而是用信息增益比率(gain ratio)，增益比率通过引入一个被称作分裂信息(split information)的项来惩罚取值较多的 feature， 分裂信息用来衡量 feature 分裂数据的广度和均匀性:

$$\text{SI}(D, f) =  -\sum\limits_{i=1}^n \frac{|D_i|}{|D|} \text{log}_2 \frac{|D_i|}{|D|} $$

$$ \text{GR}(D, f) = \frac{g(D, f)}{\text{SI}(D, f)} $$

where SI stands for Split Information, and GR for Gain Ratio.

### ID3 / C4.5 / CART
C4.5 是 ID3 算法的扩展。

1. ID3 使用信息增益，而 C4.5 使用增益比率
2. ID3/C4.5 每次划分分组时都会消耗特征，即划分数据分组之后特征数目会减少，而 CART 并不总是消耗特征
3. ID3 只能处理离散性变量，而 C4.5, CART可处理连续型变量
4. ID3/C4.5 只能用于分类任务，CART 可用于回归
5. ID3/C4.5 通过剪枝来权衡树的准确性与泛化能力，CART用全部数据发现所有可能的树结构进行对比


C4.5 较 ID3 的优势
1. 可以处理连续属性值
2. 可以处理缺失值 
    - 丢弃存在缺失值的样本
    - 补上该属性的均值或者众数 (频率最高的值)
3. 通过预剪枝来解决过拟合问题


# CART 
CART, Classification And Regression Trees, 分类决策树。采用 Gini 指数来划分属性。

优点，可以对复杂和非线性的数据建模；缺点是，结果不易理解。

CART 采用二元切分来处理连续型变量，即每次把数据集切成两份，如果数据的某特征值大于切分所要求的值，那么这些数据就进入树的左子树，反之进入树的右子树。

如何度量连续型数值的不一致度？首先计算所有数据的均值，然后计算每条数据的值到均值的差值(绝对值或者平方值)。 



# 小结

| 算法   | 支持模型 | 特征选择方法 | 连续值处理 | 缺失值处理 | 剪枝 | 树结构 |
| :---- | :-----  | :--------- | :------- | :-- | :-- | :-- |
| ID3  | 分类  | 信息增益  | No | No | No | 多叉树 |
| C4.5 | 分类  | 信息增益比	| Yes| Yes| Yes | 多叉树 |
| CART	| 分类，回归 |	基尼系数，均方差 | Yes | Yes | Yes | 二叉树 |

算法不足之处：

1. 无论是 ID3, C4.5 还是 CART，在做特征选择的时候都是选择最优的一个特征来做分类决策，但有时候分类决策由一组特征来决定，得到的决策树会更加准确。这类决策树叫做**多变量决策树(multi-variate decision tree)**。在选择最优特征的时候，多变量决策树不是选择某一个最优特征，而是选择最优的一个特征线性组合来做决策。代表算法 OC1。

# Q & A
1. 预剪枝与后剪枝如何实施，有什么区别？

预剪枝在生成树的过程中提前停止树的增长，而后剪枝是在已生成的过拟合决策树上进行剪枝。 方法是通过设定树的深度、结点中样本个数、在测试集上准确度提升值等阀值来停止决策树生长
