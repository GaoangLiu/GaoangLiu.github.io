---
layout:     draft
title:      NLP Skim Reading
date:       2023-01-03
tags:   [nlp]
categories: 
- nlp
---

# A Frustratingly Easy Approach for Joint Entity and Relation Extraction

主要贡献：
- 设计了一个简单的 end2end 关系抽取，即采取 2 个独立的 encoder 分别用于实体抽取和关系识别，在 ACE04、ACE05 和 SciERC 数据集中达到 SOTA 结果，超过了之前的使用相同 pretrain encoders 的联合抽取模型(joint model)。
- 验证了分别学习实体和关系的**上下文表征**的重要性。
- 验证了在关系提取模型中整合实体信息（边界及类型）的重要性。
- 提出了一个高效的推理方法，在牺牲一点点性能（1% drop on f1）的情况下，推理速度有 8-16 倍的提升。

<div style="display: flex; justify-content: center;">
    <img src="https://image.ddot.cc/202311/joint_entity_relation_model_20231114_0746.png" width=678pt id='joint-entity-relation-model-image'>
</div>

Pipeline 模型：
1. 实体：如[上图(a)](#joint-entity-relation-model-image)所示，采取 span-level NER 的方式，即基于片段排列的方式，提取所有可能的片段排列，通过 Softmax 对每一个 Span 进行实体类型判断。Span 的好处是可以解决嵌套实体问题，但计算复杂度较高，理论上可能的实体组合有 $n(n+1)/2$ 个，其中 $n$ 为句子长度。
2. 关系：如上图(b)所示，对所有的实体 pair 进行关系分类。其中最重要的一点改进，就是将实体边界和类型作为标识符加入到实体 span 前后，然后作为关系模型的 input。例如，对于实体 pair（Subject和Object）可分别在其对应的实体前后插入以下标识符：
   - <S:Md>和</S:Md>：代表实体类型为 Method 的 Subject，S 是实体 span 的第一个 token， /S 是最后一个 token；
   - <O:Md>和</O:Md>：代表实体类型为 Method 的 Object，O 是实体 span 的第一个 token，/O 是最后一个 token； 


计算时，对每个实体 pair 中第一个 token 的编码进行组合，然后进行 Softmax 分类，判断是否属于某一个关系，计算开销比较大。 

> 文献参考
- span-based 的 NER 模型可参考 [《Span-based Joint Entity and Relation Extraction with Transformer Pre-training》](https://arxiv.org/pdf/1909.07755.pdf)
- span-based model 边界平滑，防止模型过于自信 [《Boundary Smoothing for Named Entity Recognition》](https://arxiv.org/pdf/2204.12031.pdf)
- 多头选择 [《Joint entity recognition and relation extraction as a multi-head selection problem》](https://www.sciencedirect.com/science/article/abs/pii/S095741741830455X?via%3Dihub)
- 层叠式指针标注 [《A Novel Cascade Binary Tagging Framework for Relational Triple Extraction》](https://arxiv.org/pdf/1909.03227.pdf)
  

## FAQ <span id='joint-relation-extract-faq'></span>
- Q: Joint model 跟 end2end model 有什么区别？

- Q: 有哪些可以推广的？


# Highway Networks
论文：[《Highway Networks》](https://arxiv.org/pdf/1505.00387.pdf)

主要贡献：
- 提出了一种新的网络结构，Highway Network，可以有效地训练深层网络，缓解梯度消失或梯度爆炸的问题。如下图所示，在 MNIST 分类实验上，Highway Network 在网络层数较深时，相比于传统的深层网络，仍然可以取得很好的效果。

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/highway_networks_20231114_0752.png" width=678pt>
    <figcaption style="text-align:center"> 图. Highway Network 结构 </figcaption>
</figure>

主要思路也比较直观，即将一层的输出 $y$ 与输入 $x$ 进行叠加，并传给下一层。令 $y=H(x, W_H)$ 表示网络的上一层输出。 Highway 定义了两个非线性变换：
1. $T(x, W_T)$ 表示 transform gate，$x$ 是输入；
2. $C(x, W_C)$ 表示 carry gate。

Highway Network 的输出为定义为输出 $y$ 和输入 $x$ 的加权和，即：
$$y = H(x, W_H) \odot T(x, W_T) + x \odot C(x, W_C)$$

在实验中，$T$ 定义为 sigmoid 函数，$C$ 定义为 $1-T$，即：

$$T(x, W_T) = \sigma(W_Tx + b_T)$$

后面还有 KaiMing He 的 [《Deep Residual Learning for Image Recognition》](https://arxiv.org/pdf/1512.03385.pdf) 也是类似的思路。

He 提出了 residual learning（残差学习）的概念，即在网络中间的某一层，将输出直接连接到网络的最后一层，如下图所示。这样做的好处是，可以避免梯度消失或梯度爆炸的问题，因为在反向传播时，梯度可以直接从最后一层传到中间层，而不需要经过多层的传递。两个工作在理念上都是一样的，都是将**输出表述为输入和输入的一个非线性变换的线性叠加**。

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/skip_conn_20231114_0840.png" width=378pt>
    <figcaption style="text-align:center"> 图. residual learning 结构 </figcaption>
</figure>

差异主要在**设计思路/实现方式**上：Highway Network 是通过门控机制，控制信息的流动，参数通过学习获得，依赖于具体数据。 而 Residual Network 直接叠加输入与及非线性变换，是无参的。

这个设计带来的结果是，residual network 不需要额外的参数，训练起来更加简单；highway 更为灵活，通过学习可以获得不同的叠加权重。


