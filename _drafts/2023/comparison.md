---
layout:     post
title:      How Contextual are Contextualized Word Representations
date:       2023-09-28
tags:   [elmo, bert, embedding]
categories: 
- nlp
---

词表征有多种方式，深度学习在 NLP 领域发扬光大之前，最常用的是 one-hot 表示法，即将每一个词表示成一个向量，向量的维度等于词表的大小，这种表示向量维度很高，且向量之间没有语义关系，仅从向量上观察很难判断两个词之前有多大关系。

深度学习时代，词表征的发展比较迅速，代表方法及模型有 word2vec、GloVe、ELMo、BERT 等等。这些表征方式基于分布式假设（distributional hypothesis）的，即词的语义可以通过它的上下文来表示。得益于深度学习的强大表征能力，我们可以用（相对）低维的稠密向量来表示词。 

这种表示称为词向量或者词嵌入，关于词向量的上下文性(contextual)，Kawin Ethayarajh 等人在他们的[工作](https://aclanthology.org/D19-1006.pdf)中提出了两个概念：
- isotrapy，各向同性，对应的物理解释是，即向量独立同分布在向量空间中。 相对地，
- anisotrapy，各向异性，表示词向量（或者说词表征、词嵌入）在向量空间中集中在一个狭小的圆锥中，而不是均匀分布在整个空间中（[《The strange geometry of skip-gram with negative sampling》](https://aclanthology.org/D17-1308/)）。

在物理学中，各向异性指一个物理量在各个方向上的性质不同，比如光波在透明介质中传播时，在不同的方向上可能有不同的速度、折射率或偏振状态；再比如岩石中的地震波传播中，纵波（P波）和横波（S波）在不同方向上的传播速度和路径会有差异。

“圆锥”这个描述的不能狭隘的理解成词向量都聚集在一个“锥”内，而应该理解成**向量扎堆**。在高维空间中，各向异的向量分布是不均匀的，就像戈壁滩上的灌木丛一样，多的地方特别多，荒的地方又特别荒。以二维空间为例，假设 $Y$ 服从均匀分布 $U(1, 3)$，$X$ 服从双峰分布 $0.5 \mathcal{N}(-10, 1) + 0.5 \mathcal{N}(10, 1)$，那么 $Z = (X, Y)$ 的分布大概如下图所示：

<div style="display: flex; justify-content: center;">
  <img src="https://image.ddot.cc/202311/bimodal_distribution_rc.png" width=678pt>
</div>


再以三维空间为例，isotropic 的分布大概如下图左所示，而 anisotropic 的分布大概如下图右所示。

<div style="display: flex; justify-content: center;">
  <div style="margin-right: 10px;">
    <img src="https://image.ddot.cc/202311/firework-isotropic.jpeg_rc.png" width="300pt">
  </div>

  <div style="margin-left: 10px;">
    <img src="https://image.ddot.cc/202311/firework-anisotropic.jpeg_rc.png" width="300pt">
  </div>
</div>


那么，词向量的分布是 isotrapic 还是 anisotrapic 有什么区别呢？

作为表征的一种形式，一个好的词向量至少具备以下几个特点：
1. 语境独特性(context specific)，同一个词在不同语境下的词向量应该是不同的。显然通过 word2vec、GloVe 等方法得到的词向量是不具备这个特点的，因为这些方法都是静态的，因此是上下文无关的。
2. 语义相近（e.g., `king` and `empire`）、语法相关（e.g., `king` and `queen`）的词的词向量应该是相近的，而不相关的词的嵌入是互相远离的。 

如果词向量分布是各向异的（anisotropic），即不同词的向量在方向上都比较集中。在涉及到语义相似性的任务中，我们通常会通过余弦距离来衡量词向量的相似性，那么即使两个词有较大的语义差异，它们之间的余弦距离也会很小。那这种表示在下游任务上的表现会很差。例如，在情感分类任务中，如果“糟糕”的词向量在方向上与“好”的词向量很接近，那么“体验很好”和“体验很糟糕”情感极性也会很接近，这显然是不合理的。


怎么度量词向量的各同向性呢？[《How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings》](https://aclanthology.org/D19-1006.pdf)中提出了三个指标：
1. Self-similarity，自相似性
2. Intra-sentence similarity， 句内相似性
3. Maximum explainable variance， 最大可解释方差

# 自相似性（self-similarity）

给定一个词 $w$，有$n$条语句 $\{s_{i_1}, ..., s_{i_n}\}，$定义第 $p$ 层的 self similarity 定义为 

$\text{SelfSimilarity}_p(w) = \frac{1}{n^2-n} \sum_j \sum_{k\neq j} \cos(f_l(s_j, i_j), f_l(s_k, i_k)) $

其中， $i_j$ 表示词 $w$ 在语句 $s_j$ 中的位置。

这个指标统计的是模型生成的词向量在不同语境下的差异性。如果这个值很大，说明模型生成的词向量在不同语境下的差异性很小，也即是词向量不具有语境独特性。一种极端情况是，模型生成的词向量在不同语境下完全一样，即等同于静态词向量，这时 $\text{SelfSimilarity}_p(w) = 1$。

作者对比了 ELMo，BERT，GPT-2 模型在各层生成的词向量 self-similarity。越到模型的顶层（upper layer），self-similarity 越小，也即是词向量越具有语境独特性，这跟直觉是一致的。

在所有层中，停用词在所有单词中自相似性最低，比如，在 ELMo 中，具有平均最低自相似性的几个词分别为 "and", "of", "the" 及 "to"。作者给出的结论是：<span style="color: salmon;">驱动词的上下文表征变化的主要因素是上下文，而不是一词多义性</span>（ ... the variety of contexts a word appears in, rather than its inherent polysemy, is what drives variation in its contextualized representations）。 

但这不能说明一词多义性对这个表征变化没有影响，或者影响很少，作者没有统计 self-similarity 跟词频的关系（这里的词频指同一个词在不同上下文中出现的次数）。我的推测是，如果具有一个一词多义的词 $w_a$ 的词频跟停用词的词频一样大，那么 $w_a$ 的 self-similarity 可能会更低。

<div style="display: flex; justify-content: center;">
  <div style="margin-right: 10px;">
    <img src='https://image.ddot.cc/202311/self-similarity_rc.png' width='700pt'>
  </div>
</div>

## (An)Isotropy
关于 ELMo、BERT、GPT-2 生成的表征，作者还做了另一组对比实验，观察特定层的表征是否是 isotropic 的，即平均余弦相似度等于 0。结论是：**这些表征都是 anistropic**，即分布比较集中，且随着层数的增加，这种现象更加严重。实验结果如下图所示：

<div style="display: flex; justify-content: center;">
  <div style="margin-right: 10px;">
    <img src='https://image.ddot.cc/202311/layer-anisotropic_rc.png' width='700pt'>
  </div>
</div>

注：
- 从第 5 层开始，BERT、GPT-2 的表征的平均余弦相似度比平均自相似度还要大，这是因为在统计自相似度结果时，已经移除了 anisotropy 的影响。 
- Isotropy 跟 self-similarity 有关联，但不是一个概念。二者的区别在于，isotropy 是针对整个词表的，而 self-similarity 是针对单个词的。

# 句内相似性（intra-sentence similarity）
另一个指标是句内相似性（intra-sentence similarity），也可理解为词间相似性，度量的是一句话中不同词的词向量**是否高内聚**，也即是这些词向量是否非常相似。指标是通过计算<span style="color:blue;">每一个词的词向量与整句话的平均词向量的余弦相似度的均值</span>获得。

$\text{IntraSim}_p(s) = \frac{1}{n} \sum_i \cos(\bar{s_p}, f_p(s, i))$

where $\bar{s_p} = \frac{1}{n} \sum_i f_p(s, 1)$

理想情况下，不同词的词向量应该不同，且越不相关的词对应的词向量差异性越大，因此距离向量中心也是比较远的。这个时候 $\text{IntraSim}_p(s)$ 应该是很小的。
反之，如果 $\text{IntraSim}_p(s)$ 很大，说明通过模型学到的词向量是高度相似的，这样的词向量是不可靠的。


目前可得结论：
1. 词向量语境独特性越好（即不同语境下向量差异较大），$\text{SelfSimilarity}_p(w)$ 越小。
2. 不同词的词向量越不相关，$\text{IntraSim}_p(s)$ 越小。

那么 1， 2 反过来成立吗？

答案是“未必”。假设一个模型把每一个句中的每一个词都映射到向量空间的不同方向，甚至同一语境下同一个词也都映射到不同的方向，那么 $\text{SelfSimilarity}_p(w)$ 就会很小，但这样的向量表示也是没有意义的。比如说，同义词之间的相似性就会很低，如 “car” 和 “automobile”。


# 最大可解释方差（maximum explainable variance）

最后一个指标是最大可解释方差（maximum explainable variance），度量的是一个词的词向量在不同语境下的变化程度。
给定层数 $p$、单词 $w$，令 $M=[f_p(s_1, i_1), ..., f_p(s_n, i_n)]$ 表示 $w$ 的 occurrence matrix，令 $\sigma_1, ...,\sigma_m$ 表示 $M$ 的前 $m$ 个奇异值，那么最大可解释方差定义为：

$\text{MEV}_p(w) = \frac{\sigma_1}{\sum_{i=1}^n \sigma_i}$

我们知道，奇异值对应着矩阵中隐含的重要信息，且重要性和奇异值大小正相关。直观上讲，这个值就是用来衡量$w$所有表征中“最好”的那个表征所占的比重。换句话说，确定$w$，如果要用一个静态向量代替所有向量，那么这个向量能以多大概率“胜任”这个任务。这个值越小，说明这个词的词向量在不同语境下的变化越大，就越不太可能用一个静态向量代替所有向量。


｜ 关于奇异值物理意义的解释，参考 [奇异值分解的物理意义是什么？](https://zhuanlan.zhihu.com/p/42896542)

<div style="display: flex; justify-content: center;">
  <div style="margin-right: 10px;">
    <img src='https://image.ddot.cc/202311/mev_rc.png' width='700pt'>
  </div>
</div>


# 小结 
在 ELMo、BERT、GPT-2 这三个模型中，
1. 任意一层的表征都是各向异性的。
2. 随着层数的增加，自相似性越来越小，也即是同一个词的词向量越具有语境独特性。
3. 这些模型生成的表征的 MEV 只有不到 5%，即静态表征只能表达这些表征 5% 的信息，再次验证了模型生成的表征确实比静态表征更好。

一句话总结：模型生成的表征确实比静态表征更能体现语境信息，但也不是完美的。 
