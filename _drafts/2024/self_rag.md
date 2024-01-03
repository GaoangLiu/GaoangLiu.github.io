---
layout:     post
title:      RAG 优化方案一 —— SELF-RAG
date:       2024-01-03
tags:   [rag, embedding]
categories: 
- nlp
---


RAG 通过引入相关的检索段落来增强大语言模型（LLMs）的输入，从而在知识密集型任务中减少事实错误。然而，不管检索到的信息是否有助于任务，vanilla RAG 都会把检索结果输入到 LLM，这可能会限制LLMs的灵活性，并导致生成的结果质量较差。




[Akari Asai 等](https://arxiv.org/abs/2310.11511)对 RAG 做了一个优化，提出**自我反思检索增强生成**（Self-Reflective Retrieval Augmented Generation, SELF-RAG），通过**按需检索**及**自我反思**来提高 LLM 的生成质量。

SELF-RAG 通过端到端的方式训练一个 LM，在给定任务输入的情况下生成任务输出以及间歇性的特殊标记（即反思标记）。这些反思标记分为**检索标记**和**评论标记**，这两个标记分别用于指示检索的必要性以及生成质量（见图1右侧）。特别的，在给定输入提示和之前的生成文本的情况下，SELF-RAG首先判断引入检索能够改善生成效果。如果可以，它将输出一个检索标记，按需调用检索模型（步骤1）。接着，SELF-RAG同时处理多个检索到的段落，评估它们的相关性，然后生成相应的任务输出（步骤2）。最后，它生成评论标记，对自己的输出进行打分？，以事实准确性和整体质量为标准选出最佳的输出（步骤3）


<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202401/self-rag-vs-rag_20240103_0943.png" width=789pt>
    <figcaption style="text-align:center"> 图 1. Self-RAG 概览 </figcaption>
</figure>

系统分为两个部分，一个生成器$\mathcal{M}$，一个判别器 $\mathcal{C}$。

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202401/self-rag-algorithm_20240103_1000.png" width=789pt>
    <figcaption style="text-align:center"> SELF-RAG 算法 </figcaption>
</figure>

# Q & A
1. 怎么实现的？
2. 效果如何？
3. 为什么有效？
4. 代价是什么？

## 5. 数据如何获取的?


# 参考
- Akari Asai et al., [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511), arXiv, 2023.10.17




