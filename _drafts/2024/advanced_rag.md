---
layout:     draft
title:      RAG 再探
date:       2024-01-01
tags:   [rag, embedding]
categories: 
- nlp
---



在之前的文章[《RAG 初探》]({{site.baseurl}}/2023/11/16/Retrivial-augmented-generation/)里，我们浅聊了 RAG 的基本原理，vanilla RAG 的工作方式概述下来说是：将文本切块，然后使用 transformer encoder 向量化，将向量放入索引。在用户发起一个 query 时，我们把 query 用同一个 encoder 向量化，然后跟索引中的向量做相似度计算，得到 top-K 个文档，最后将这些文档及 query 送入 LLM 里，让 LLM 生成答案。

用户能支配的就只有 query，不同场景下，这个 query 可以是一个问题，比如“what is a black hole”，或者一个 prompt。

整个 pipeline 的结构如下图所示：

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202312/rag_structure_20231231_1052.png" width=789pt>
    <figcaption style="text-align:center"> RAG pipeline 的主要组成部分 </figcaption>
</figure>

大概可以分为切片+向量化、索引、检索、生成几个部分。切片+向量化这一步没有什么好说的，需要注意的是文档切片的长度与语义完整性要保持合适的平衡，切的太碎，语义就没了。 向量化这一块可以从 [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) 找个合适的开源方案。 

# 索引 
## 创建索引 
索引有很多成熟的工具：[faiss](https://faiss.ai/)，[nmslib](https://github.com/nmslib/nmslib)，[annoy](https://github.com/spotify/annoy)。LlamaIndex 也支持许多[向量存储索引](https://docs.llamaindex.ai/en/latest/community/integrations/vector_stores.html)。

## 索引优化 
**层次索引**：如果文档比较多，考虑创建两个索引：一个摘要索引，一个文档块索引。搜索时两步走，先通过摘要筛选出相关文档，然后仅在这个相关组内进行搜索。

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202312/hierarchinal_rag_20231231_1208.png" width=786pt>
    <figcaption style="text-align:center"> 层次索引 </figcaption>
</figure>

另一个思路是为每一个 chunk 都用 LLM 生成一个相关的问题，然后向量化并存储这些问题，搜索时先将 query 与问题做匹配，然后将匹配到的问题对应的 chunk 送入 LLM 生成答案。


# 搜索 
## 混合/融合搜索 
把基于关键字（比如 TFIDF、BM25）的搜索结果与基于向量的搜索结果组合起来。 这种方法的关键一环是把具有不同相似度分数的检索结果正确融合起来，毕竟 BM25 的分数跟向量的相似度分数具备不同的物理意义。
一个办法是借助[倒排融合算法（reciprocal rank fusion,RRF）](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)来解决，对检索结果进行重新排序以获得最终输出。

$$RRF(d\in D)=\sum_{r\in R} \frac{1}{k+r(d)}$$

RRF 的思路是，对于每一个文档 $d\in D$，计算它在每一个检索结果中的排名 $r(d)$，然后将这些排名加权求和，权重是 $\frac{1}{k+r(d)}$，其中 $k$ 是一个超参数，用来扼制异常高排名的影响。这个方案的一个优点是，它不需要知道检索结果的具体分数，只需要知道排名就可以了。

# Query 改造
有些 query 比较复杂，比如 “LangChain, LlamaIndex 在 Github 上哪个 star 更多”，这时候就需要对 query 进行拆分，变成两个简单的问题 “LangChain 在 Github 上的 star 数是多少” 以及 “LlamaIndex 在 Github 上的 star 数是多少”，然后将两个问题的答案相减，得到最终的答案。这个过程叫做 query 变换（query transformation）。


# 响应合成 
RAG pipeline 的最后一步，是将检索到的文档及 query 送入 LLM 生成答案。简单粗暴的做法是所有检索召回的上文都拼接到一起，然后跟 query 一起丢给 LLM。精细一点的做法也有，比如：
1. 先 summarize 检索到的文档，然后跟 query 一起丢给 LLM。
2. 根据不同的 context chunks 生成多个不同的答案，在答案的基础上做 summary。
3. 一次只给 LLM 一个 context chunk，迭代的优化答案。


refer: https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6




