---
layout:     post
title:      Dimensionality Reduction
date:       2020-01-10
tags: [pca, machine learning, dimensionality reduction]
categories: 
- machine learning
---

-- 主要途径：projection, Manifold Learning
-- 主要技术：PCA, Kernal PCA, LLE 

# Projection 

# 流形学习
流形学习基于流形假设，即很多高维数据集 lie close to 低维的流形。

## Principle Component Analysis
PCA 从数据中识别其主要特征，通过沿着数据**最大方差**方向旋转坐标轴来实现。

奇异值分解 (Singular Value Decomposition, SVD)


### 使用 Scikit-Learm
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x2d = pca.fit_transfrom(X) # or 

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transfrom(X)
```
方法 1 显式的将维度降为 2；方法 2 则选择最小的维度 $$d$$ 使得这些维度的 variance ration 加和至少为 0.95 。







# 问题 
1. 降维的主要动机是什么？有什么缺陷？
    1. 加快算法训练速度 
    2. 便于可视化及审查数据最主要特征
    3. 节省空间
    缺陷：
    1. 损失信息
    2. 增加计算负担
    3. 降维后的特征通常不直观，难以解释
