---
layout:     post
title:      Active Learning
date:       2022-12-10
tags: [active learning]
categories: 
- deeplearning
---


Goal: 通过主动选择最有价值的样本进行标注的机器学习方法。其目的是使用尽可能少的、高质量的样本标注使模型达到尽可能好的性能。

### 分类

按应用场景分类

- membership query synthesis
- stream-based
- pool-based

### 查询策略

意义：判断样本的价值，是否值得标注。

1. 不确定性采样 uncertainty sampling。挑最不确定的，即当前模型觉得最困难的来标注，可以用熵等表示这种不确定性。
2. 多样性采样 diversity sampling。根据数据分布以保证diversity。方法有
    1. 代表性采样
3. 预期模型更改 expected model change
4. 委员会查询 query by committee, QBC. 多个模型组成的委员会投票，选择分歧最大的样本

### 问题

1. 性能不稳定。
    1. 策略和数据样本是影响性能的两在关键因素。
        1. 数据层面：对非常冗余的数据，比 random 效果好，但对多样性强，冗余性低的样本的数据集，效果比 random 差。
2. 迁移困难
    1. 不同域、不同任务之间。 

# 框架
[modAL](https://modal-python.readthedocs.io/en/latest/)，一个基于 Python3 的 AL 框架。在 sklearn 的基础上进行了封装，提供了一些基本的 AL 算法。目前支持的 AL 策略有:
- uncertainty sampling: [least confident](https://www.sciencedirect.com/science/article/pii/B978155860335650026X?via%3Dihub), max margin and max entropy
- committee-based algorithms, vote entropy, consensus entropy and max disagreement
- 多标签策略 multi-label strategies: SVM binary minimum, max loss, mean max loss
- 期望误差减少 expected error reduction: binary and log loss
- Bayesian optimization: probability of improvement, expected improvement and upper confidence bound 
- batch active learning: ranked batch-mode sampling


# 参考

- 主动学习概述及最新研究 2022.01.13 [https://www.cvmart.net/community/detail/6018](https://www.cvmart.net/community/detail/6018)
- Active Learning in Context of Natural Language Processing – Hands On https://www.inovex.de/de/blog/active-learning-in-context-of-natural-language-processing-hands-on/
- Active learning, 综述 https://burrsettles.com/pub/settles.activelearning.pdf
