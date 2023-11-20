---
layout:     post
title:      Application of Contrast Learning to Text Representation
date:       2022-11-28
tags: [nlp, contrast learning]
categories: 
- nlp
--- 

# 表征学习
文本表征学习就是将一段文本映射到低维向量空间，获取句子的**语义表示**，大致经历过四个阶段：

- 统计类型，典型的方法是利用 TD-IDF 抽取关键词，用关键词表示表征整个句子。
- 深度模型阶段，此阶段方式较多，自从 glove、word2vec 等词粒度的表示出现后，在此基础有比较多的延伸工作，从对句子中的词向量简单平均、到有偏平均 SIF[1]，后来引入 CNN、LSTM 等模型利用双塔、单塔方式进行学习句子表示，比较典型的几个工作有
    - 微软在排序搜索场景的 DSSM[2]，将 word 进行 hash 减少词汇个数，对 word 的表示进行平均得到句子原始表达，经过三层 MLP 获取句子表示。
    - 多伦多大学提出的 Skip-Thought[3]，是 word2vec 的 skip-ngram 在句子表达的延伸，输入三个句子，用中间一句话，预测前后两句话。
    - IBM 的 Siam-CNN[4]，提出了四种单塔、双塔不同的架构，利用 pairwise loss 作为损失函数。
    - Meta 的 InferSent[5]，在双塔的表示基础上，增加了充分的交互。

- Bert、Ernie 等预训练大模型阶段，在此阶段比较基础典型的工作有：
    - 由于 Bert 通过 SEP 分割，利用 CLS 运用到匹配任务场景存在计算量过大的问题，Sentence-BERT[6] 提出将句子拆开，每个句子单独过 encoder，借鉴 InferSent 的表示交互，来学习句子表达。
- 20 年在图像领域兴起的对比学习引入到 NLP。

# 对比学习
对比学习（contrast learning）一般划分到无监督学习（USL）的范畴，典型范式就是：<span style="color:blue">[代理任务](https://stats.stackexchange.com/questions/404602/pretext-task-in-computer-vision)+目标函数</span>，这两项也是对比学习与有监督学习（SL）最大的区别。


SL 中有输入 $x$，有对应的 ground truth $y$，计算模型输出的 $y_p$ 与 $y$ 通过目标函数计算损失，指导模型训练。对于 USL 来说，是没有 ground truth 的，而这里就是代理任务发挥作用的地方，代理任务用来定义正负样本，我们通过训练一个模型来解决代理任务，从而学习到一个好的表征，使得这个表征可以轻松适应到下游任务。[SimCLR](https://arxiv.org/pdf/2002.05709.pdf) 的框架如下图，大体流程：
1. 数据增强，采样，构造正负样本。
2. 通过对比损失训练特征提取器 （encoder） $f$ 及一个映射头 $g$，$g$ 用来将 $f$ 的输出映射到一个低维空间。在 SimCLR 中 projection head 是一个两层的 MLP，维度是 128 维。
3. 在下游任务中，把 projection head 去掉，只保留 $f$，用 $f$ 的输出作为特征，进行下游任务的训练。

<figure class="half">
    <img src="https://image.ddot.cc/202311/simclr_20231120_2017.png" width="700" />
    <figcaption>SimCLR framework</figcaption>
</figure>




# 在 NLP 中应用
对比学习的目标是使得**相似的东西表示越相似，不相似的东西越不相似**。一般训练过程：

1. 通过数据增强的方式构造训练数据集，对于一条数据，数据集需要包含正例（相似的数据）和负例（不相似的数据）。
   1. 增强方式如，term 替换、随机删除、回译等 
2. 将正例和负例同时输入到 encoder 模型中。
3. 最小化正例之间的距离，最大化负例之间的距离，进行参数更新。

在语义相似度任务中，一种基于对比学习的方法是 [SimCSE]({{site.baseurl}}/2022/10/18/Semanticv-Similarity/)。

1. 损失联合方式自监督：将 CL 的 loss 和其他 loss 混合，通过联合优化，使 CL 起到效果：CLEAR，DeCLUTER，SCCL。
2. 非联合方法自监督：构造增强样本，微调模型：Bert-CT，ConSERT，SimCSE

CLEAR
Paper: https://arxiv.org/pdf/2012.15466.pdf



https://www.51cto.com/article/681705.html

# 参考
- [1] [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx)
- [2] [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)
- [3] [Skip-Thought Vectors](https://arxiv.org/pdf/1506.06726.pdf)
- [4] [Siamese Recurrent Architectures for Learning Sentence Similarity](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00051)
- [5] [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/pdf/1705.02364.pdf)
- [6] [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf)
- [7] [Contrastive Learning for Sentence Similarity](https://arxiv.org/pdf/2004.11362.pdf)
- [8] [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf)
- [9] https://zhuanlan.zhihu.com/p/584195919
  
