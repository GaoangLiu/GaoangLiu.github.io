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


ChatGPT 爆火之后，有一段时间内很多公司都在竞相做向量数据库，一些数据库厂商也在竞相在传统数据库上增加向量存储功能，支持相似度检索。常规做法是通过预训练获取一个大模型，然后将数据向量化并存储。这种做法有几个问题：



1. 幻觉问题。这是 LLM 的通病之一 。 
2. 无法更新、扩展。模型一旦训练好，就无法更新了，而且更新模型的成本很高。
3. 可解释性差。难以解释解释为什么这个向量与这个文档相关。

知识难以更新的问题对于一些信息动态更新的任务来说，是一个巨大的挑战。例如，对于一些 QA 任务，答案可能会随着时间的推移而变化，e.g., “最近一届奥运会在哪个城市举办的？”，这时候就需要更新模型的知识，传统的做法对于这一问题就无能为力了。LLM 在近两年的发展中，一个被广泛认可的技术是 RAG （Retrieval-Augmented Generation）。这个技术由 meta 于 2020 年在论文 [《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》](https://arxiv.org/pdf/2005.11401.pdf) 中提出，它是一个检索增强的生成模型，通过检索得到的上下文信息来指导生成，从而提高生成的质量。这里面比较关键的一环是“检索增强”，检索用到的知识可以轻松地随时更改或补充，在需要更新知识时，不需要重新训练整个模型。RAG 使用的检索方案是 DPR(Dense Passage Retrieval)，即是 20 年推出的“暴打前浪 BM25” 的技术，同样也是 meta 的工作。DPR 整体结构是一个 dual encoder: document encoder 和 query encoder，两个 encoder 使用的模型都是 $$\text{BERT}_\text{BASE}$$。关于 DPR 的机制，我们在之前的文章[《Okapi-BM25》]({{site.baseur}}/2022/11/17/Okapi-BM25/)里稍有提过。另一个模块是 seq2seq 生成器，模型使用的 [BART-large](https://arxiv.org/abs/1910.13461)（也是 meta 的工作）。


# RAG 结构 

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/rag_20231116_0836.png" width=678>
    <figcaption style="text-align:center"> RAG 结构 </figcaption>
</figure>

给定输入为 $$x$$，retriever $$p_\eta(x)$$ 召回 top-K 个 $$z_i$$，($$z_i$$ 形成一个分布，按概率选择一个，有点像 sampling？)，然后生成器 $$p_\theta(y_i\lvert x, z, y_{1,...,i-1})$$ 逐 token 生成 $$y_i$$。

比较重要的一点，RAG 将被检索文档视为潜变量，即 $$z$$ 是隐变量，这样可以使用变分推断来训练模型。模型有两种结构，一个是 RAG-sequence，一个是 RAG-token。RAG-sequences 在生成一条序列的每个 token 时，使用是同一个文档，而 RAG-token 在生成每一个 token 时，使用的是不同的文档。

$$\begin{aligned}
p_\text{RAG-sequence}(y\lvert x) &\approx \sum_{z\in \mathcal{Z}} p_\eta(z\lvert x)p_\theta(y\lvert x, z) \\\
&= \sum_{z\in \mathcal{Z}} p_\eta(z\lvert x)\prod_{i=1}^N p_\theta(y_i\lvert x, z, y_{1,...,i-1})
\end{aligned}$$

其中，$$\mathcal{Z}= \{z_1, ..., z_K\}$$ 是检索器召回的 top-K 个文档，$$p_\eta(z\lvert x)$$ 是检索器的输出，$$p_\theta(y_i\lvert x, z, y_{1,...,i-1})$$ 是生成器的输出。

对应的 RAG-token 在生成每一个 token 时都有一个对应的边际分布，训练模型也即是最大化下面的概率：

$$\begin{aligned}
p_\text{RAG-token}(y\lvert x) &\approx \prod_{i=1}^N \sum_{z\in \mathcal{Z}} p_\eta(z\lvert x)p_\theta(y_i\lvert x, z_i, y_{1,...,i-1})
\end{aligned}$$

## Retriever 
在 extractive QA 任务中，答案存在于语料一个或多个段落中，答案对应于段落的一个 span。DRP 的做法是先将文档拆分成等长的段落，然后再从段落中提取 span。
形式化的，假设语料由 $$D$$ 个文档构成，$$d_1, d_2, ..., d_D$$，拆分后得到 $$M$$ 个段落，$$ \mathcal{C} = \{ p_1, p_2, ..., p_M\}$$，其中每一个 $$p_i$$ 可以视为一个 token 序列 $$p_i = \{w_{i,1}, w_{i,2}, ..., w_{i,|p_i|}\}$$。那么给定一个问题 $$q$$，retriever 的目标是从 $$\mathcal{C}$$ 中检索出最相关的 $$K$$ 个段落，即：$$R: (q, \mathcal{C}) \rightarrow \mathcal{C}_\mathcal{F}$$，其中 $$\lvert \mathcal{C}_\mathcal{F}\rvert = K \ll M$$。


# 训练 
RAG 联合训练检索模块跟生成模块，不需要关于检索文档的监督信息（这也是为什么文中说将 document 视为潜变量），在给定数据集 $$\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^N$$ 的情况下，最小下面的负对数似然：

$$\begin{aligned}
\mathcal{L}(\theta, \eta) &= - \sum_{(x, y)\in \mathcal{D}} \log p_\text{RAG}(y\lvert x) \\\
&= - \sum_{(x, y)\in \mathcal{D}} \log \sum_{z\in \mathcal{Z}} p_\eta(z\lvert x)p_\theta(y\lvert x, z)
\end{aligned}$$

在训练过程，由于更新检索库的编码器消耗巨大，因为每更新一次文档编码器就需要对所有文档重新编码，所以在训练过程 RAG 选择固定文档编码器 $$\text{BERT}_d$$ 的参数，只训练 $$\text{BERT}_q$$ 编码器与 BART 生成器。 

# 效果

在 4 个 open-domain QA 数据集，都达到了 SotA。在 abstractive QA 上全面超过了 BART，且比较接近 SotA。只是接近 SotA 也能说明效果？ 

RAG 在 MSMARCO NLG 等任务上训练时，对比 SotA 并没有使用 golden passage，仅使用了问题与答案。一部分问题的答案在没有 golden passage 的情况下不可能推断出来，依赖于 wikipedia 的知识（RAG 的 non-parametric memory）就更不行了。RAG 只能依靠生成器（BART）的生成能力去获取答案。

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/rag_result_20231117_2131.png" width=778>
    <figcaption style="text-align:center"> RAG 实验结果 </figcaption>
</figure>




# 变分推断 

变分推断（Variational Inference，VI）是一种近似推断方法，将后验推断问题转化为一个优化问题。它的思想是将后验分布 $$p(z\lvert x)$$ 近似为一个简单的分布 $$q(z)$$，然后通过最大化变分下界（Evidence Lower Bound，ELBO）来近似后验分布。一般流程是确定一个分布簇 $$\mathcal{Q}$$，然后在这个分布簇中找到一个分布 $$q(z)$$，使得 $$q(z)$$ 与 $$p(z\lvert x)$$ 的 KL 散度最小，即：

$$q^*(z) = \arg\min_{q(z)\in \mathcal{Q}} KL(q(z)\lvert\lvert p(z\lvert x))$$

这个分布簇既要足够灵活，可以拟合复杂的后验分布，又要足够简单，方便优化。

那为什么要用 VI 呢，有没有其他方法可以做？

后验推断要估算的是 $$p(z\lvert x)$$，根据贝叶斯公式，有：

$$p(z\lvert x) = \frac{p(x\lvert z)p(z)}{p(x)}$$

其中，$$p(x\lvert z)$$ 是似然函数，$$p(z)$$ 是先验分布，$$p(x) = \int p(x\lvert z)p(z) dz = \int p(x,z) dz$$ 是关于潜变量的 $$z$$ 的边际似然，也称 evidence。对很多模型来说， $$p(x)$$ 没有解析解，或者即使有，计算复杂度也很高。VI 可以通过优化变分下界来近似后验分布，从而不需要直接计算 $$p(x)$$。当然也有其他方法，例如 MCMC，但 MCMC 的计算复杂度高，而且收敛速度慢。关于 VI 如何近似后验分布，可以参考博客 [变分推断(Variational Inference)初探](https://www.cnblogs.com/song-lei/p/16210740.html)。


## 参考 
- [NeurIPS 2020|RAG：为知识密集型任务而生](https://zhuanlan.zhihu.com/p/264485658)
- [变分推断（Variational Inference）进展简述](https://zhuanlan.zhihu.com/p/88336614)
- [《Variational Inference: A Review for Statisticians》](https://arxiv.org/pdf/1601.00670.pdf)。
- [变分推断(Variational Inference)初探](https://www.cnblogs.com/song-lei/p/16210740.html)