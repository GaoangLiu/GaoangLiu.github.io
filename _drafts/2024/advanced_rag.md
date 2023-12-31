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


refer: https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6




