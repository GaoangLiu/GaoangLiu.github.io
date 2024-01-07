---
layout: post
title: Probability
date: 2022-10-16
tags: math probability
categories: math
author: gaonagliu
---
* content
{:toc}


# Metrics
## KL Divergence
KL （[Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) ）散度 （相对熵，信息差）是衡量两个概率分布 $$P, Q$$ 之间差异的一种度量方法。KL 散度的定义为：




$$D_{KL}(P\lVert Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}$$
> 如果 $$P, Q$$ 为连续分布，那么 $$D_{KL}(P \lVert Q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx$$
> 
以上定义刻画的是当我们用 $$Q$$ 来表示 $$P$$ 时，需要多少额外的信息。一般来说，$$P$$ 代表观察到的数据或者一个真实分布，$$Q$$ 代表模型分布，因此 $$D_{KL}(P \lVert Q)$$ 也被称为模型复杂度。

KL 散度满足的性质：
- 非负性。如果 $$P$$ 和 $$Q$$ 是相同的分布，那么 $$D_{KL}(P \lVert Q) = 0$$。
- 非对称性，即 $$D_{KL}(P \lVert Q) \neq D_{KL}(Q \lVert P)$$
