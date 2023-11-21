---
layout:     post
title:      Loss Functions
date:       2022-02-01
tags: [loss]
categories: 
- nlp
---

DL 中很多任务都涉及到 softmax 计算，比如多分类、语言模型等。以语言模型为例，softmax 的需要计算每一个词的概率，复杂度是 $O(V)$，其中 $V$ 是词汇表大小。对于大规模的词汇表，计算量是非常大的，因此有很多方法用来近似 softmax，比如 hierarchical softmax、negative sampling、noise contrastive estimation 等。

# Sparse softmax <span id='sparse-softmax'> </span>

用于需要输出一个 sparse output 的场景，比如推荐系统中，只保留一些最相关的选项。原作[《From Softmax to Sparsemax:A Sparse Model of Attention and Multi-Label Classification》](https://arxiv.org/pdf/1602.02068.pdf)称可增强解释性并提升效果。 参考苏老师设计一个[版本](https://spaces.ac.cn/archives/8046/comment-page-2):

||orginal|sparse|
|---|---|---|
|softmax|$$p_i=\frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}$$|$$p_i=\begin{cases}\frac{e^{x_i}}{\sum_{j \in \Omega_k} e^{x_j}}& i \in \Omega_k \\0& i \not \in \Omega_k\end{cases}$$|

其中 $\Omega_k$ 是将 $x_1, x_2, ..., x_n$ 从大到小排序后的前 $k$ 个元素的下标集合。思路是，计算出来结果后，只保留前 $k$ 个最大的概率，其余的概率置为 0。$k$ 是一个超参数，$k=n$时，等价于原始的 softmax。

Torch 版本的一个实现参考[Github](https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py)。


# Hierarchical softmax <span id='hierarchical-softmax'> </span>
为解决 softmax 计算复杂度高的问题，hirarchical softmax 尝试建立一个二叉树或 Huffman树，将词汇表分成多个子集，每个子集对应二叉树的一个叶子节点，每个叶子节点对应一个词，内部节点对应一个概率。这样就将 softmax 的计算复杂度从 $O(V)$ 降低到 $O(\log V)$。



# Contrastive Loss

## Noise Contrastive Estimation
噪声对比估计（NCE）是一种用来估计 softmax 的方法，通过负采样的方式，将 softmax 的计算复杂度从 $O(V)$ 降低到 $O(K)$，其中 $V$ 是词汇表大小，$K$ 是负采样的个数。基本思想是将一个多分类问题转成一个二分类问题。
