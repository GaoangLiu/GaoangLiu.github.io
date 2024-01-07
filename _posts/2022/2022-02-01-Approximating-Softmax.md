---
layout: post
title: Approximating Softmax
date: 2022-02-01
tags: embedding
categories: nlp
author: gaonagliu
---
* content
{:toc}


DL 中很多任务都涉及到 softmax 计算，比如多分类、语言模型等。Softmax 的一个问题是计算复杂度高，以语言模型为例，softmax 需要计算每一个词的概率，复杂度是 $$O(V)$$，其中 $$V$$ 是词汇表大小。对于大规模的词汇表，计算非常困难。这个问题催生多种解决方案，用来近似 softmax，比如 hierarchical softmax、negative sampling、noise contrastive estimation 等。




下面以语言建模为例，假设序列 $$T = w_1, ..., w_T$$，词汇表为 $$V$$。给定上下文  $$c$$，词 $$w$$ 的概率为：

$$\begin{aligned}
p(w\lvert c) &= \frac{h^T v'_w}{\sum_{w_i \in V} h^T v'_{w_i}} \\\
&= \frac{\exp(h^T v'_w)}{\mathcal{Z}(c)}
\end{aligned}$$

其中 $$h$$ 是倒数第二层的输出，$$v'_w$$ 是词 $$w$$ 的 embedding，令 $$d$$ 表示 embedding 的维度，$$v'_w \in \mathbb{R}^d$$，$$h \in \mathbb{R}^d$$。
计算 $$\mathcal{Z}(c)$$ 需要对词汇表中的每一个词都计算一次 $$h^T v'_w$$，复杂度是 $$O(V)$$，这个计算量对于大规模的词汇表来说是非常大的。

# Hierarchical softmax <span id='hierarchical-softmax'> </span>
[Hierarchical softmax(H-Softmax)](https://proceedings.neurips.cc/paper_files/paper/2008/file/1e056d2b0ebd5c878c550da6ac5d3724-Paper.pdf) 本质是将平铺的 softmax 层以二叉树的形式组织起来，每个叶子节点对应一个词，内部节点对应一个概率。



# Sparse softmax <span id='sparse-softmax'> </span>

用于需要输出一个 sparse output 的场景，比如推荐系统中，只保留一些最相关的选项。原作[《From Softmax to Sparsemax:A Sparse Model of Attention and Multi-Label Classification》](https://arxiv.org/pdf/1602.02068.pdf)称可增强解释性并提升效果。 参考苏老师设计一个[版本](https://spaces.ac.cn/archives/8046/comment-page-2):

||orginal|sparse|
|---|---|---|
|softmax|$$p_i=\frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}$$|$$p_i=\begin{cases}\frac{e^{x_i}}{\sum_{j \in \Omega_k} e^{x_j}}& i \in \Omega_k \\0& i \not \in \Omega_k\end{cases}$$|

其中 $$\Omega_k$$ 是将 $$x_1, x_2, ..., x_n$$ 从大到小排序后的前 $$k$$ 个元素的下标集合。思路是，计算出来结果后，只保留前 $$k$$ 个最大的概率，其余的概率置为 0。$$k$$ 是一个超参数，$$k=n$$时，等价于原始的 softmax。

Torch 版本的一个实现参考[Github](https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py)。




# Contrastive Loss

## Noise Contrastive Estimation
噪声对比估计（NCE）是一种用来估计 softmax 的方法，基本思想是将**从词表中预测某个词的多分类问题，转为从噪音词中区分出目标词的二分类问题**，一类是数据类别 data sample，另一个类是噪声类别 noisy sample，通过学习数据样本和噪声样本之间的区别，将数据样本去和噪声样本做对比，也就是“噪声对比（noise contrastive）”，从而发现数据中的一些特性。但是，如果把整个数据集剩下的数据都当作负样本（即噪声样本），虽然解决了类别多的问题，计算复杂度还是没有降下来，解决办法就是**做负样本采样来**计算 loss，这就是 estimation 的含义，也就是说它只是估计和近似。一般来说，负样本选取的越多，就越接近整个数据集，效果自然会更好。

以语言模型为例，训练的目标是最小化每一个词的交叉熵，即：

$$\begin{aligned}
\mathcal{J}(\theta) &= - \log \frac{\exp(h^T v'_w)}{\sum_{w_i \in V} \exp(h^T v'_{w_i})} \\\
&= - h^T v'_w + \log \sum_{w_i \in V} \exp(h^T v'_{w_i})
\end{aligned}$$

NCE loss 的一般形式为：
$$\begin{aligned}
\mathcal{L}_{\text{NCE}_k} &= \sum_{(w,c) \in \mathcal{D}} \log p(D=1\lvert w, c) + k \mathbb{E}_{\tilde{w} \sim P_n(w)} \log p(D=0\lvert \tilde{w}, c) \\\
&= \sum_{(w,c) \in \mathcal{D}} \log \frac{p(w\lvert c)}{p(w\lvert c) + k \sum_{\tilde{w} \in \mathcal{V}} p(\tilde{w}\lvert c)} + k \sum_{(w,c) \in \mathcal{D}} \log \frac{k p(\tilde{w}\lvert c)}{p(w\lvert c) + k \sum_{\tilde{w} \in \mathcal{V}} p(\tilde{w}\lvert c)}
\end{aligned}$$


## InfoNCE
Info NCE loss是NCE的一个简单变体，它认为如果你只把问题看作是一个二分类，只有数据样本和噪声样本的话，可能对模型学习不友好，因为很多噪声样本可能本就不是一个类，因此还是把它看成一个多分类问题比较合理，公式如下：

$$\begin{aligned}
\mathcal{L}_q = -\log \frac{\exp(q^T k_+ / \tau)}{\exp(q^T k_+ / \tau) + \sum_{k \in \mathcal{K}} \exp(q^T k / \tau)}
\end{aligned}$$

# 参考
- [On word embeddings - Part 2: Approximating the Softmax](https://www.ruder.io/word-embeddings-softmax/)