---
layout: post
title: Conditional Random Field
date: 2021-07-15
tags: crf
categories: algorithm
author: GaoangLau
---
* content
{:toc}


# 条件随机场
concepts
1. 属于对数线性模型; 




设有联合概率分布 𝑃(𝑌) ，由无向图 𝐺=(𝑉,𝐸) 表示，在图 𝐺 中，结点表示随机变量，边表示随机变量之间的依赖关系，如果联合概率分布 𝑃(𝑌) 满足成对、局部或全局马尔可夫性，就称此联合概率分布为**马尔可夫随机场**（Markov random filed）也称**概率无向图模型**（probablistic undirected graphical model）：

- 成对马尔可夫性：设𝑢,𝑣是无向图𝐺中任意两个没有边连接的结点，其他所有结点表示为𝑂，对应的随机变量分别用𝑌𝑢,𝑌𝑣,𝑌𝑂表示，成对马尔可夫性是指给定随机变量组𝑌𝑂的条件下随机变量𝑌𝑢,𝑌𝑣是条件独立的，如下：

    𝑃(𝑌𝑢,𝑌𝑣|𝑌𝑂)=𝑃(𝑌𝑢|𝑌𝑂)𝑃(𝑌𝑣|𝑌𝑂)
- 局部马尔可夫性：设𝑣∈𝑉是𝐺中任意一个节点，𝑊是与𝑣有边连接的所有节点，𝑂是𝑣，𝑊以外的其他所有节点。𝑣表示的随机变量是𝑌𝑣，𝑊表示的随机变量是𝑌𝑤，𝑂表示的随机变量是𝑌𝑜。局部马尔可夫性是在给定随机变量组𝑌𝑤的条件下随机变量𝑌𝑣与随机变量𝑌𝑜是独立的。

    𝑃(𝑌𝑣,𝑌𝑂|𝑌𝑊)=𝑃(𝑌𝑣|𝑌𝑤)𝑃(𝑌𝑂|𝑌𝑊)



read this series
1. crf https://www.cnblogs.com/en-heng/p/6214023.html
2. memm 最大熵马尔可夫模型 https://www.cnblogs.com/en-heng/p/6201893.html
3. hmm https://www.cnblogs.com/en-heng/p/6164145.html

# References
https://www.cnblogs.com/xlturing/p/10161840.html    