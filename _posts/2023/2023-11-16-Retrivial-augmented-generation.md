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

ChatGPT 爆火之后，有一段时间内很多公司都在竞相做向量数据库，一些数据库厂商也在竞相在传统数据库上增加向量存储功能。之前在一个 AI 讲座上，一个向量数据库公司的老板也在侃侃而谈趋势、痛点、机遇，在当时看来，向量数据库几乎是 GPT 浪潮下一个不可多得的风口。

但多数人的注意力还是聚焦在大模型上，向量数据库的关注逐渐式微，关于检索，一个比较引入注目的技术是 RAG （Retrieval-Augmented Generation）。这个技术与 meta 于 2020 年在论文 [《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》](https://arxiv.org/pdf/2005.11401.pdf) 中提出，它是一个检索增强的生成模型，通过检索得到的上下文信息来指导生成，从而提高生成的质量。这里面有两个主要的模块，一个是检索，使用的技术是 DPR(Dense Passage Retrieval)，即是 20 年出的暴打前浪 BM25 的技术，同样也是 meta 的工作，这个丹琦大佬也有参与。关于 DPR 的结构，我们在之前的文章[《Okapi-BM25》]({{site.baseur}}/2022/11/17/Okapi-BM25/)里稍有提过。另一个模块是 seq2seq 生成器，模型使用的 [BART](https://arxiv.org/abs/1910.13461)（也是 meta 的工作）


# RAG 结构 

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/rag.svg" width=678>
    <figcaption style="text-align:center"> RAG 结构 </figcaption>
</figure>





