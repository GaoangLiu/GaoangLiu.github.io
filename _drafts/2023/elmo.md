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

