---
layout:     post
title:      ELMo
date:       2023-11-06
tags:   [elmo, bert, embedding]
categories: 
- nlp
---

之前在文章[《How Contextual are Contextualized Word Representations》]({{site.baseurl}}/2023/09/28/How-Contextual-are-Contextualized-Word-Representations/)里，我们稍微介绍了词嵌入的上下文性(contexuality)，并侧重讨论了词嵌入的各向同性（isotrapy）。一个各向同性的词嵌入在高维空间中是独立同分布的，这是一个“好”的词表征的必要条件。

经典的词表征方式例如 one-hot 有很大的局限性，比如它的维度很高，且向量之间缺少语义关系，仅从向量上几乎无法区分相近词与语义相斥词的差异，一词多义性更不用说。 word2vec(2013) 及后续的 GloVe(Global Vectors for Word Representation， 2014)通过密集的词向量替代 one-hot 矩阵，解决了高维稀疏、缺少语义关联的问题，但这些方法都是静态的、上下文独立的，不能处理一词多义的问题。作为 [NNAC 2018 outstanding papers](https://naacl2018.wordpress.com/2018/04/11/outstanding-papers/) 之一的 ELMo，（[《Deep contextualized word representations》](https://arxiv.org/pdf/1802.05365.pdf)）则提出了一个比较好的解决思路。

# 网络结构 

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/elmo_20231115_0809.webp" width=678>
    <figcaption style="text-align:center"> ELMo 结构 (image source: BERT) </figcaption>
</figure>

- $E$ 是词嵌入层，将词 $w$ 映射到一个 $d$ 维的向量 $x_w$；
- 中间两层是双向 LSTM，分别是前向 LSTM 和后向 LSTM，它们的输出分别是 $h^{fwd}$ 和 $h^{bwd}$；
- $T$ 是最终的词表征。 

ELMo 基于双向语言模型（biLM）的思路，通过训练一个双向语言模型，得到一个双向的词向量表征，一个词的向量表征不仅仅依赖于它自身，还依赖于它的上下文。特点：
1. **双向 LSTM 建模**：ELMo 使用双向 LSTM 来学习上下文相关的表示，同时考虑了从左到右和从右到左的上下文信息。对于给定的输入序列（例如句子），ELMo 会经过前向 LSTM 和反向 LSTM 两次，产生两组隐状态。
2. **加权组合多层向量**： ELMo 使用加权组合不同层的表示来生成最终的词表示。这些权重是通过训练数据学习得到的。对于每个词，ELMo 会计算一个权重向量，然后将不同层的表示按权重相加。因此 ELMo 在生成词嵌入时更灵活地考虑不同层次的信息。
3. **上下文感知表示**： 由于 ELMo 是在整个句子上训练的，因此它能够捕捉到每个词的上下文信息。这意味着相同的词在不同上下文中可能有不同的表示，从而更好地捕捉了词汇的多义性和上下文相关性。在 [How Contextual are Contextualized Word Representations](https://arxiv.org/abs/1909.00512) 一文中，作者通过实验表明，ELMo 在第二层的 self-similarity 已经非常接近 BERT 最后一层的 self-similarity（后者更低，越低越好）且比 BERT 第二层的低，说明 ELMo 在词嵌入的上下文性还是比较不错的。

# 模型训练 

给定一个由 $N$ 个词组成的句子 $s = \{t_1, ..., t_N\}$，ELMo 中 biLM 的目标是最大化下式：
$$
\sum_{k=1}^N(\log p(t_k|t_1, ..., t_{k-1}; \Theta_{x}, \Theta_f, \Theta_{s}) + \log p(t_k|t_{k+1}, ..., t_{N}; \Theta_{x}, \Theta_b, \Theta_{s}))
$$

其中，$\Theta_{x}$ 是词嵌入层的参数，$\Theta_f$ 和 $\Theta_b$ 分别是前向 LSTM 和后向 LSTM 的参数，$\Theta_{s}$ 是 softmax 层的参数。

对于每一个 token $t_k$，biLM 中对应的向量有 $2L+1$ 个，

$$
V_k = \{x^k, h_{f1}^k, ..., h_{fL}^k, h_{b1}^k, ..., h_{bL}^k\}
$$

其中 $x^k$ 是词嵌入层的输出，$h_{f1}^k, ..., h_{fL}^k$ 是前向 LSTM 的输出，$h_{b1}^k, ..., h_{bL}^k$ 是后向 LSTM 的输出，$L$ 是 LSTM 的层数。

ELMo 将 $V_k$ 中的向量按照权重相加，得到最终的词向量表征 $R_k$：

$$
R_k = \gamma \sum_{j=0}^{2L}s_j h_j^k
$$

其中，$s_j$ 是 softmax 层的输出，$\gamma$ 是一个标量，用于缩放 $R_k$ 的大小。


# 模型使用
主要有两种，一种是将其 ELMo 作为下游任务的输入，例如将通过 ELMo 获得的词向量作为分类器的输入。另一种方式是将 ELMo 作为预训练模型，然后在下游任务中微调。


