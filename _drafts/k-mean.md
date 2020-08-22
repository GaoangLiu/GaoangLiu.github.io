---
layout:     post
title:      K-Mean Clustering 
date:       2020-07-28
tags: [machine learning, k-mean]
categories: 
- machine learning
---

基本思想：通过迭代方法寻找$$K$$个簇的一种划分方案，使得聚类结果对应的代价函数最小。特别的，代价函数定义为*各个样本距离所属簇中心点的误差平方和*
$$
\begin{aligned}
    J(c, \mu) = \sum\limits_{i=1}^M \| x_i - \mu_{c_i}\|^2
\end{aligned}
$$
$$\mu_{c_i}$$代表簇的中心点，$$M$$样本总数。

## K-Mean 一般步骤
1. 数据预处理，如归一化、去除离群点等
2. 随机选取$$K$$个中心
3. 定义代价函数 $$J(c, \mu) = \underset{\mu}{\text{min}}\underset{c}{\text{min}} \sum\limits_{i=1}^M \| x_i - \mu_{c_i}\|^2$$
4. 迭代
    1. 对每一个样本，将其分配到距离最近的簇
    2. 对每一类簇，重新计算中心(距离和平均)

## 特点
1. 优点
    1. 相对高效，计算复杂度 $$ O(NKt) $$，其中 $$ t $$是迭代轮数。

2. 缺点
    1. 易受初值(初始中心点)及离群点的影响，结果不稳定
    2. 结果通常是局部最优，而不是全局最优(NP-Hard)
    3. 簇颁差别比较大(如一类是另一类的 100 倍)情况下无法很好聚类
    4. 不太适合离散分类 ？？？
    5. 需要人工预先确定 $$ K $$ 值，与真实 $$ K $$ 未必一致


Tips: 
1. NP-Hard，所有 NP 问题都可以约化到这类问题，但这类问题未必是 NP 问题。但如果它也是 NP 问题，那么称为 NPC (NP Complete，NP 完全) 问题。NPC 是 NP-Hard 与 NP 的交集。

## 归一化与离群点处理
Why ?
1. 本质上是基于$$ L_p$$ 度量的数据划分方法，$$L_p$$ 度量的值由数值最大的分量主导。因此均值与方差的维度将对数据的聚类结果产生决定性的影响，做归一化为了消除这种影响。
2. 离群点及噪声数据会对均值产生较大的影响，导致中心偏离


## 合理选择 K 值
这是 K 均值聚类的主要问题之一，方法:
1. 手肘法。$$ K$$ 越大，损失函数越小。从一个小的 $$K$$ 开始不断增大 $$K$$ 并计算，直到损失函数的值变化幅度减小
```python
for k in range(2, 10): 
    model = KMeans(n_clusters=k)
    model.fit(dataset)
    sse.append(model.inertia_)
plt.plot(range(2, 10), sse)
```
2. 轮廓系数(Solhouette Coefficient)
3. Gap statistic 方法。定义为 $$ \text{Gap}(K) = E(\log D_k) - \log D_k $$，其中
    * $$ E(\log D_k) $$ 是 $$ \log D_k $$ 的期望，一般通过 Monte-Carlo 求出
    * 物理意义，随机样本的损失与实际样本的损失差
    * $$ \text{Gap}(K) $$ 最大的 $$ K$$ 即为最佳的簇数
4. 采用核函数

### 轮廓系数
结合了内聚度与分离度两个因素
* 计算样本 $$i$$ 到同簇其他样本的平均距离 $$ d_a^i $$，称为 **簇内不相似度**
* 计算样本$$i$$到其他簇的样本的最短距离 $$ d_b^i = \text{min} \{ d_1, ..., d_{k-1}\}$$，称为**簇间不相似度**
* 定义的轮廓系数 $$s_i = \frac{d_b^i - d_a^i}{\text{max}\{d_a^i, d_b^i\}}$$

将所有样本点的轮廓系数取平均，即得到聚类结果的轮廓系数。

Examples with Sklearn
```python
# Search for the best K
from sklearn.metrics import silhouette_score
silhouette_coefficients = []
for k in range(2, 10): # At least two clusters required
    model = KMeans(n_clusters=k)
    model.fit(dataset)
    score = silhouette_score(dataset, model.labels_)
    silhouette_coefficients.append(score)
plt.plot(range(2, 10), silhouette_coefficients)
```


## K-Mean 的改进
1. K-Mean ++，对初始点的选择方式进行改进
    1. 先随机一个簇中心点 $$ u_1 $$，这与 K-mean 一致
    2. 选择第 $$ 1 < l \leq K $$ 个中心点时，距离前 $$l-1$$ 个聚类中心越远的点会有更高的概率被选中
    
2. ISODATA (Iterative Self Organizing Data Analysis Technique yAy!) algorithm，迭代自组织分析法
    * 思想上，当一个类别的样本数过少，则去除该类别; 当一个类别的样本数过多、分散程度较大，则拆分该类别
    * 优点，不必预先指定 $$K$$ 值，只需给一个参考值。
    * 缺点，参数多
        * 类别样本最少数目
        * 类别最大方差，用以定义分散程度
        * 类别之间最小距离 


## KNN v.s. K-Mean
KNN 是分类算法，属于监督学习的范畴。$$ K $$ 的含义是特征空间中定义新样本类别的最近的样本的个数，

K-mean 聚类算法，属于无监督学习。$$ K $$ 的含义是想要划分的簇的个数。
