---
layout: post
title: Retrivial augmented generation
date: 2023-11-16
tags: rag embedding
categories: nlp
author: berrysleaf
---
* content
{:toc}


本文关注以下几个问题 。
- [ ] RAG 是什么？
- [ ] 如何实现的？



- [ ] 有什么优势 ？
- [ ] 使用场景有哪些？

ChatGPT 爆火之后，有一段时间内很多公司都在竞相做向量数据库，一些数据库厂商也在竞相在传统数据库上增加向量存储功能。常见的做法是通过预训练获取一个大模型，然后将数据向量化并存储。


关于检索，一个比较引入注目的技术是 RAG （Retrieval-Augmented Generation）。这个技术与 meta 于 2020 年在论文 [《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》](https://arxiv.org/pdf/2005.11401.pdf) 中提出，它是一个检索增强的生成模型，通过检索得到的上下文信息来指导生成，从而提高生成的质量。这里面有两个主要的模块，一个是检索，使用的技术是 DPR(Dense Passage Retrieval)，即是 20 年出的暴打前浪 BM25 的技术，同样也是 meta 的工作，这个工作丹琦大佬也有参与。关于 DPR 的结构，我们在之前的文章[《Okapi-BM25》]({{site.baseur}}/2022/11/17/Okapi-BM25/)里稍有提过。另一个模块是 seq2seq 生成器，模型使用的 [BART](https://arxiv.org/abs/1910.13461)（也是 meta 的工作）。


# RAG 结构 

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/rag_20231116_0836.png" width=678>
    <figcaption style="text-align:center"> RAG 结构 </figcaption>
</figure>

给定输入为 $$x$$，retreiver $$p_\eta(x)$$ 召回 top-K 个 $$z_i$$，($$z_i$$ 形成一个分布，按概率选择一个，有点像 sampling？)，然后生成器 $$p_\theta(y_i\lvert x, z, y_{1,...,i-1})$$ 逐 token 生成 $$y_i$$。

比较重要的一点，RAG 将被检索文档视为潜变量，即 $$z$$ 是隐变量，这样可以使用变分推断来训练模型。

## 变分推断 
变分推断（Variational Inference，VI）是一种近似推断方法，将后验推断问题转化为优化问题。具体地，它通过最大化变分下界（Variational lower bound）来近似后验分布。变分推断的思想是将后验分布 $$p(z\lvert x)$$ 近似为一个简单的分布 $$q(z)$$，然后通过最大化变分下界来近似后验分布，即：
$$
\begin{aligned}
\log p(x) &= \log \int p(x, z) dz \\\
&= \log \int q(z) \frac{p(x, z)}{q(z)} dz \\\
&\geq \int q(z) \log \frac{p(x, z)}{q(z)} dz \\\
&= \mathcal{L}(x, q)
\end{aligned}$$

其中 $$x$$ 是观测数据，$$z$$ 是潜变量，$$q(z)$$ 是简单的分布，$$p(x, z)$$ 是联合分布。$$\mathcal{L}(x, q)$$ 是变分下界，也称为 ELBO（Evidence Lower BOund）。简单来说，VI 就是用简单的分布 $$q(z)$$ 来近似复杂的后验分布 $$p(z\lvert x)$$。这个分布 $$q(z)$$ 的要求？？？ 关于 VI，一个比较好的教程是 [《Variational Inference: A Review for Statisticians》](https://arxiv.org/pdf/1601.00670.pdf)。

后验推断要估算的是 $$p(z\lvert x)$$，根据贝叶斯公式，有：

$$p(z\lvert x) = \frac{p(x\lvert z)p(z)}{p(x)}$$

其中，$$p(x\lvert z)$$ 是似然函数，$$p(z)$$ 是先验分布，$$p(x) = \int p(x\lvert z)p(z) dz$$ 是？？？。 




