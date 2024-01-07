---
layout:     post
title:      Attention is all you need - reread
date:       2022-09-30
tags:   [skip-gram, embedding]
categories: 
- nlp
---


再读 [《Attention Is ALL You Need》](https://arxiv.org/pdf/1706.03762.pdf)。之前关于 transformer 的理解，都是从 [《The Illustrated Transformer》](http://jalammar.github.io/illustrated-transformer/) 这篇文章开始的，自己班门弄斧写了一个很简单的[Transformer]({{site.baseurl}}/2022/10/06/Transformer-Encoder/)笔记，写的含糊且单薄，觉得有必要再重读一下经典，重新理解一下 transformer。

# 背景 
作者在提出 transformer 之前，RNN、LSTM、GRU 大行其道，科研圈自不必说，在 BERT 出现之前，Kaggle 比赛里的主流深度模型基本上都是在 LSTM 基础上调深度、调结构。 BERT 出现之后，所有人的“注意力”都转移到了 transformer 上，传统的 RNN 方案都是拿来做 baseline，SOTA 模型基本上都是 transformer。当然，现在 LLM 热闹起来，又有了新的 baseline。比如 Kaggle 2023 [LLM-science-exam](www.kaggle.com/competitions/kaggle-llm-science-exam) 比赛中，前几名都是微调 LLM 的方案，像 deberta 这样的模型也只能做为一个 base（详细解读参考公众号文章[大模型Kaggle比赛首秀冠军方案总结](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ%3D%3D&mid=2650439341&idx=2&sn=f93f18a5c0c74899d912a69e811ddabe)）。

[LLM](https://arxiv.org/abs/2304.13712) 百花齐放，但其基础仍然是 transformer 结构，可以预测，在未来的一段时间内，transformer 将仍然是 NLP 领域的主流模型。

回到发展历史，RNN 通常按照输入和输出序列的符号位置进行计算。将位置与计算时间步骤对齐，在每一时间步$t$生成隐藏状态$h_t$的序列，作为下一个时间步隐藏状态$h_{t+1}$的输入的函数。这种固有的顺序特性使得在训练过程中无法并行化训练样本，这在序列长度较长时会比较棘手，因为内存约束限制了跨示例的批处理（... memory constraints limit batching across examples）。

另一方面，注意力机制可以将任意两个位置上的符号关联起来，从而可以解决 RNN 中的长距离依赖问题。当然，由于注意力机制通常是序列长度的二次函数，因此在序列长度较长时，计算成本会很高。

# 结构 
整体上仍然是 [encoder-decoder](https://arxiv.org/pdf/1406.1078.pdf) 的结构，encoder 将输入输入序列 $(x_1, ..., x_n)$ 映射到连续的表示 $z=(z_1, ..., z_n)$。Decoder 从 $z$ 中生成输出序列 $(y_1, ..., y_m)$。每个步骤都是一个自回归模型，如下图所示。

<div style="display: flex; justify-content: center;">
  <div style="margin-right: 10px;">
    <img src='https://image.ddot.cc/202311/transformer-1001.png' width='500pt'>
  </div>
</div>

代码实现及实验见博客 [《transformer implementation》]({{site.baseurl}}/2023/01/06/Transformer-implemenation/)。

# Attention 
关注两个概念，一个是 self-attention，一个是 multi-head attention。

## Scaled Dot-Product Attention
Scaled Dot-Product Attention(以下简称 SDPA) 是 transformer 中的核心，也是 transformer 的基本组成单元。Scaled Dot-Product Attention 的计算公式如下：

$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V $

SDPA 是计算 attention 权重的一种方式，计算流程大致是，首先计算 query 和 key 的内积，然后除以 $\sqrt{d_k}$，最后通过 softmax 函数得到 attention 权重，再将 attention 权重与 value 相乘得到最终的 attention 输出。这里的 $d_k$ 是 query 和 key 的维度。
除了 scaled dot production attention 之外，还有很多其他的计算方式，参考[《An Attentive Survey of Attention Models》](https://arxiv.org/pdf/1904.02874.pdf)。

<div style="display: flex; justify-content: center;">
  <div style="margin-right: 10px;">
    <img src='https://image.ddot.cc/202311/alignment-functions.png' width='500pt'>
  </div>
</div>


关于上面的设计有几个小问题需要考虑，第一个，为什么要除以 $\sqrt{d_k}$ 呢？

这里除以一个 $\sqrt{d_k}$ 一是为了**稳定梯度**，保证注意力分数的值在一个合理的范围之内，有助于防止梯度爆炸或梯度消失问题，从而提高模型的训练稳定性。 二是为了**控制关注范围**，因为 query 和 key 的维度 $d_k$ 越大，那么 query 和 key 的内积的值就越大，softmax 函数的输出就越接近于 0 或 1，这样就会导致 attention 的权重分布不均匀，即只关注某些特定的位置，而忽略其他位置的信息。通过除以缩放因子，可以确保注意力权重不受查询维度大小的影响。

第二个问题，为什么要用 softmax 函数？有没有其他的选择？

这里的 softmax 实际上在做归一化，保证注意力权重的和为 1 。当然也有其他的归一化方法，比如 [sparsemax]({{site.baseurl}}/2023/01/01/NLP-concepts/)，Gumbel Softmax 等等。 

第三个问题，为什么要用 query 和 key 的内积作为 attention 权重？物理意义是什么？

注意力本质上是一种**加权分配机制**，允许模型在处理输入数据时有选择性地关注不同的部分，而不是一概而论地对所有输入进行处理。作为一种**相似度度量**，内积可以用来衡量 query 和 key 之间的相关性，给相关性高的位置分配更大的权重，从而使模型更加关注这些位置。

## 多头 
作者发现，与其使用一个 $d_\text{model}$(e.g., 512) 维的单头注意力，不如使用 $h$ 个 $d_k$(e.g., 64) 维的多头注意力，这样可以让**模型同时关注不同位置的信息，从而提高模型的表征能力和学习能力**。多头注意力的计算公式如下：

$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$

其中，$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$，$W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$，$W_i^K \in \mathbb{R}^{d_{model} \times d_k}$，$W_i^V \in \mathbb{R}^{d_{model} \times d_v}$，$W^O \in \mathbb{R}^{hd_v \times d_{model}}$。

多头是通过将输入的特征进行线性变换（对应上面的$W_i^Q, W_i^K, W_i^V$），获得单独的 $K, Q, V$，然后将这些 $K, Q, V$ 传入到多个注意力头中，最后将多个注意力头的输出拼接起来，再通过一个线性变换得到最终的输出。

## Positional encoding
Transformer 没有使用 RNN 或者 CNN，因此无法利用序列中的位置信息。为了利用序列中的位置信息，Transformer 使用了 Positional encoding。Positional encoding 的计算公式如下：

$PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{model}})$

$PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d_{model}})$

之前的文章[《Positional Encoding》]({{site.baseurl}}/2022/09/30/Positional-Encoding/) 稍微详细的介绍了一下 positional encoding，这里就不再赘述了。


# FAQ
## 什么是多头，为什么要用多头？

自注意力机制允许模型在处理输入序列时分别关注不同位置的信息，而多头自注意力机制则是在多个子空间或视角下执行这种关注操作，以增强模型的表征能力和学习能力。除了可以学习不同的关注重点及表征能力增强之外，MHA 还可以并行计算，加快训练速度。

## 为什么要用残差连接？

[Residual connections](https://arxiv.org/abs/1512.03385)主要的一个用处是缓解梯度消失问题。更多是技术上的原因而不是设计时就考虑的，参考[StackExchange:Why are residual connections needed in transformer architectures](https://stats.stackexchange.com/questions/565196/why-are-residual-connections-needed-in-transformer-architectures)。$\square$


## 为什么要用 [layer normalization](https://arxiv.org/pdf/1607.06450.pdf)？
一是加速收敛，二是提高模型的泛化能力。
模型中的每一层都包含了多个自注意力机制和前馈神经网络（feed-forward neural network），这些层之间存在输入和输出的连接。在模型训练的过程中，每一层的输入分布可能会发生变化，使得不同层之间的输入分布差异较大，这会导致模型收敛速度减慢。
Layer Normalization 的作用是对每一层的输出进行归一化操作，使得每一层的输入分布保持一定的稳定性。具体来说，对于每一层的输出 $x$，Layer Normalization 会计算其均值 $\mu$ 和方差 $\sigma$，然后对输出进行归一化操作，得到归一化后的输出 $\hat{x} $，如下所示：

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

其中，$\mu$ 和 $\sigma$ 分别是输出 $x$ 的均值和标准差，$\epsilon$ 是一个小的常数用于防止除以零的情况，$\gamma$ 和 $\beta$ 是可学习的缩放因子和偏移因子。这个归一化过程能够使得每一层的输出具有类似的分布，从而加速收敛速度和提高模型的泛化能力。


## 为什么要用 layer normalization 而不是 batch normalization？

作者并没有解释，但知乎上问题[transformer 为什么使用 layer normalization，而不是其他的归一化方法？](https://www.zhihu.com/question/395811291)下有不少有说服力的回答。

从结果说，正如苏老师所言，**估计就是实验效果比较好**。没有说非 LN 不可。 

从正则化的角度来说，LN 是对每个样本的特征进行正则化，BN 是对每个 batch 的特征进行正则化。简单来说，深度学习里的正则化方法就是**通过把一部分不重要的复杂信息损失掉，以此来降低拟合难度以及过拟合的风险，从而加速了模型的收敛**。Normalization 目的就是让分布稳定下来（降低各维度数据的方差）。不同正则方法操作的信息维度不一样。 

NLP 领域中不同 batch 样本的信息关联性不大，比如 NLI 数据集中，不同样本之前涉及的主题可能都不一致。且由于不同的句子长度不同，强行归一化会损失不同样本间的差异信息，所以就没在 batch 维度进行归一化，而是选择 LN，只考虑的句子内部维度（单样本的不同特征）的归一化。
这一点在工作[《PowerNorm: Rethinking Batch Normalization in Transformers》](https://arxiv.org/pdf/2003.07845.pdf)里也有论证，作者指出 NLP 数据中的一个 batch 的方差（相对 CV）都比较大。


## Feed forward 为什么要用两层？有没有其他的选择？

自注意力子层用于捕捉序列中的长距离依赖关系，而 feed forward 子层则用于增强模型的非线性能力与表示能力。它包含两个线性变换及一个非线性函数 ReLU，其计算公式如下：

$$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$$

第一层扩大感受野，扩展每个位置的表示，为学习更复杂的特征提供可能性，激活函数能帮助模型学习更复杂的非线性特征，然后用第二线性层将每个位置的表示压缩回原始维度。

代码实现 [Github](https://github.com/gaoangliu/gaoangliu.github.io/blob/master/codes/2023/position_wise_feed_forward.py)。


## 对比传统的 encoder-decoder 结构，transformer 有什么优势？
1. 并行计算：传统的 encoder-decoder 模型在解码过程中，每个词的生成都依赖于上一个词的结果，导致无法进行并行计算。而Transformer 引入了自注意力机制来计算每个词与所有词的关联度，因此可以并行计算每个位置的注意力权重，从而提高了计算效率。
2. 长距离依赖建模：RNN-based 的 encoder-decoder 模型往往受制于固定长度的上下文窗口，难以有效地捕捉长程依赖关系。而Transformer 通过自注意力机制，可以在每个位置上对整个输入序列进行关联性计算，从而更好地建模长程依赖关系。
3. 训练稳定：通过引入残差网络与层标准化，有助于缓解梯度消失问题，使训练更加稳定。
4. 多头：能够同时关注不同位置的输入信息，可以允许模型在不同的抽象层次上进行建模，从而更好地捕捉输入的多种关联特征。


## Decoder 中的 masked multi-head attention 有什么作用？为什么要用 masked multi-head attention？

主要目的是在训练过程中防止模型访问到未来信息。这是为了确保模型在生成输出时只依赖于先前生成的内容，遵循从左到右逐步解码的约束。

## 为什么要用 [label smoothing](https://arxiv.org/pdf/1512.00567.pdf)？
对模型进行正则化，防止模型过拟合。在 Transformer 中，标签平滑通过在计算交叉熵损失函数时，使用一个平滑参数 $\epsilon$ 来调整目标类别的标签，使其接近于 $1-\epsilon$，同时将其他类别的标签设为 $\epsilon$/(类别数-1)。 这种标签平滑的修改可以减少模型对于过度自信的预测，从而提高模型在其他样本上的泛化能力。虽然对 PPL 有负作用，但提高了模型的准确性。 

# 小结 
介绍了 transformer 的背景、模型结构、attention 机制及 position encoding。简单探讨这些组件的实现方式及意义。 


