---
layout:     post
title:      RAG 的优化方案们
date:       2024-01-03
tags:   [rag, embedding]
categories: 
- nlp
---

RAG 通过引入相关的检索段落来增强大语言模型（LLMs）的输入，从而在知识密集型任务中减少事实错误。然而，不管检索到的信息是否有助于任务，vanilla RAG 都会把检索结果输入到 LLM，这可能会限制LLMs的灵活性，并导致生成的结果质量较差。


# SELF-RAG
[Akari Asai 等人](https://arxiv.org/abs/2310.11511)对 RAG 做了一个改进，提出**自我反思检索增强生成**（Self-Reflective Retrieval Augmented Generation, SELF-RAG），通过**按需检索**及**自我反思**来提高 LLM 的生成质量。

这个方法本质是对 RAG 的 retrival + generation 过程做了细化，每走一步，都检索一些新的知识（不做检索也可以认为是没有检索到信息时的一种特殊情况），筛选后继续生成，迭代直到整个流程结束。 
这种优化思路其实很常见，比如在 prompt tuning 里面，tree of thoughts(ToT) 其实也是在 CoT 的基础上做了类似的改进，每阶段都往前走一小步，然后让模型去判断生成的结果是否对当前任务有帮助，如果有帮助则继续生成，如果没有，则探索其他路径。具体细节可参考笔者之前关于 [《Tree of Thoughs》]({{site.baseurl}}/2023/12/10/Tree-of-Thoughts/) 的学习探讨文章。

ToT 称这些中间结果为 thoughts，SELF-RAG 则称为 segments，**自我反思**做的事情就是筛选好的 segments，做为下一步的垫脚石。 

## 聊聊它的基本原理 
SELF-RAG 通过端到端的方式训练一个 LM，在给定 query 的情况下生成一段内容，以及一些特殊标记，即反思标记。这段内容（segment）可以是一句话，也可以一个段落，粒度可以调整。反思标记有两大类：**检索标记**和**评论标记**，前者用于指示**是否有必要进行检索**，后者用于对**检索召回文档的相关性、生成的 segment 与文档的知识一致性、segment 对 query 的有用性**进行打分。

具体一点，在给定输入和之前的生成文本的情况下，SELF-RAG 首先判断是否需要引入检索以改善生成效果。如果需要，它输出一个检索标记，按需调用检索模型。接着，SELF-RAG同时处理多个检索到的段落，分别评估它们的相关性，然后生成相应的任务输出。最后，它生成评论标记，对自己的输出进行评估，以事实准确性和整体质量为标准选出最佳的输出。


<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202401/self-rag-vs-rag_20240103_0943.png" width=789pt>
    <figcaption style="text-align:center"> 图 1. Self-RAG 概览 </figcaption>
</figure>

## 举个例子解释一下

举个栗子，假设query 是 $x$="How did US states get their names?"，生成器$\mathcal{M}$根据这个query生成一个 segment $y_{< t}=$“US states got their names from a variety of sources.”。

这时候把 $(x, y_{< t})$ 当成输入，再交给$\mathcal{M}$，让它判断是否需要检索外部知识，$\mathcal{M}$输出 `yes,no,continue`中的一个，如果返回 `yes` 则进行检索。如果返回 `no` 则继续生成，如果返回 `continue` 则继续使用上一次的检索结果。上面这个 query 一看就很考验历史知识，因此，检索一下外部知识是有必要的。

检索返回若干文档 $D=\{d_1,...,d_n\}$后，对每一个文档 $d$，$\mathcal{M}$判断是否跟$x$相关，即输入$(x, y_{\leq t}, d_i)$，输出 `relevant,irrelevant`中的一个。如果返回 `relevant` 则使用这个文档，如果返回 `irrelevant` 则不使用这个文档。之后，$\mathcal{M}$还会结合文档$d$生成下一句话 $y_{t}$，再判断$d$是否支持生成的$y_t$，结果有三种可能：`fully supported, partially supported, not supported`，然后还要打分判断$y_t$是否对解决$x$有帮助，结果从1分到5分，分数越高越好。最终输出结果是一个三元组： `(is-relevant, is-supported, usefulness)`。这一步里面的多个文档可以并行处理。

上面这一套操作的目的是为了筛选最合适的$y_t$，方案是根据输出结果排序，按照**召回文档的相关性、生成的 segment 与文档的知识一致性、segment 对 query 的有用性** 的标准选出实力最强的生成结果$y_t$，然后继续迭代生成。

那问题来了，如果有多个$y_t$水平相当怎么办？

<div style="overflow: hidden; border-radius: 10px;text-align: center;">
  <img src="https://host.ddot.cc/good-question-pooja-hegde.gif" alt="Your GIF" style="width: 234px; border-radius: 20px;">
</div>

首先，$y_t$的打分来源模型生成特定token（e.g., `relevant`, `not supported`）的概率归一值（参考下图2），这些值加起来，刚好存在两个 $f(y_t,d,...)$ 值相等的概率非常小。其次，即使$f(y_t,d,...)$值相等，可以通过 beam-search 的思路来保持多条路径，最终选择一条合适的，实际上这也是文中实验中使用的方案。当然，暴力一点也可以考虑贪心算法，每次随机选一个。

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202401/self-rag-score_20240104_1047.png" width=789pt>
    <figcaption style="text-align:center">图 2. 生成结果的的打分公式 </figcaption>
</figure>


整个算法流程如下：
<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202401/self-rag-algorithm_20240103_1000.png" width=789pt>
    <figcaption style="text-align:center"> 图 3. SELF-RAG 算法 </figcaption>
</figure>




# Q & A
1. 怎么实现的？
2. 效果如何？
3. 为什么有效？
4. 代价是什么？
5. 用的什么模型？



# 参考
- Akari Asai et al., [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511), arXiv, 2023.10.17




