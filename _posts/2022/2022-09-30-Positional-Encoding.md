---
layout: post
title: Positional Encoding
date: 2022-09-30
tags: nlp transformer
categories: nlp
author: gaonagliu
---
* content
{:toc}


Positional Embedding 的概念很早就已经提出了，比如在 [《Convolutional Sequence to Sequence Learning》](https://arxiv.org/pdf/1705.03122.pdf) 中提取了将绝对位置信息添加到输入中。在 [transformer]({{site.baseurl}}/2022/09/30/Attention-is-all-you-need-reread/) 被广为人知之后，大家对位置编码的关注度也越来越高。




<div style="display: flex; justify-content: left;">
    <img src='https://file.ddot.cc/imagehost/2023/position_embeddings_source.png' width='500pt'>
</div>




# 为什么需要位置编码？
Attention 模块无法捕捉输入顺序，无法区分不同位置的 token。对语言来说，句子中词汇的顺序和位置都是非常重要的。它们定义了语法，从而定义了句子的实际语义。在 Transformer 架构中，句子中的 token 流经 Transformer 的编码器、解码器堆栈，模型本身对每个单词没有任何**位置信息**的。
如果一个模型没有位置信息，那么它将**无法理解句子中单词的顺序**。例如，"Tom likes apple, but hates orange" 和 "Tom hates orange, but likes apple" 两句话的意思是完全不同的。如果模型没有位置信息，那么它将无法理解这两句话的区别。
因此，仍然需要一种方法将单词的顺序整合到模型中。想给模型一些位置信息，一个方案是在每个单词中添加一条关于它在句子中位置的信息。

针对这个问题，粗略的讲，有两个选择：
1. 绝对位置编码 (absolute PE)：将位置信息加入到输入序列中，相当于引入索引的嵌入。比如[Sinusoidal](https://arxiv.org/pdf/1706.03762.pdf), Learnable, FLOATER, Complex-order, RoPE。
2. 相对位置编码 (relative PE)：通过微调自注意力运算过程使其能分辨不同 token 之间的相对位置。比如XLNet, T5, DeBERTa, URPE。论文 [《Self-Attention with Relative Position Representations》](https://arxiv.org/pdf/1803.02155.pdf)）。


# 绝对位置编码 
## Sinusoidal Positional Encoding
在论文 [《Attention is All You Need》](https://arxiv.org/pdf/1706.03762.pdf) 中使用了三角函数进行位置编码。 

<img src="https://machinelearningmastery.com/wp-content/uploads/2022/01/PE2.png" width=789pt>

作者对 learned position embedding 和 sinusoidal position encoding 进行了对比实验，[结果](https://file.ddot.cc/imagehost/2023/transformer-pe-variations.png)没有明显差异。出于对[长度外推性](https://arxiv.org/pdf/2108.12409.pdf)和参数量规模的考虑，最终选择了 sinusodial 版本。


## Transformer 的位置编码层

给定长度为 $$L$$ 的序列，第 $$k$$ 个符号的位置编码为 $$PE(pos, 2i)=\sin(pos/10000^{2i/d_{model}})$$ 和 $$PE(pos, 2i+1)=\cos(pos/10000^{2i/d_{model}})$$，其中 pos 是实体的位置，$$i$$ 是维度的索引，$$d_\text{model}$$ 是嵌入层的维度。

在位置编码中除以 $$10000^{2i/d_{model}}$$ 是为了在位置编码的计算中引入一个缩放因子，帮助控制位置编码的数值范围，确保位置编码中的数值在不同维度上具有相对不同的变化率。


### 编码实现 
```python
import numpy as np
def get_position_encoding(seq_len:int, d:int, n=10000):
    """ Get position encoding for a sequence of length seq_len and embedding dimension d.
    """
    pe = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            pe[k, 2 * i] = np.sin(k / denominator)
            pe[k, 2 * i + 1] = np.cos(k / denominator)
    return pe
```


# 相对位置编码(RPE, Relative Positional Encoding)
并不直接建模每一个输入标记的位置信息，而是通过相对位置编码来建模。相对位置编码的思想是，每个输入标记的位置信息可以通过它与其他标记的相对位置来表示。例如，对于一个长度为 $$L$$ 的序列，第 $$k$$ 个标记的位置信息可以通过它与第 $$k-1$$ 个标记的相对位置来表示。

[《Self-Attention with Relative Position Representations》](https://arxiv.org/pdf/1803.02155.pdf) 的做法是计算 attention score 和 weighted value 时各加入一个可训练的表示相对位置的参数，并且 multi head 之间可以共享。

$\begin{aligned}
    z_i = \sum_{j=1}^L \alpha_{ij} (x_j W^V + \gamma_{ij}^V)
\end{aligned}$

其中 
$$\begin{aligned} \alpha_{ij} = \frac{\exp{e_{ij}}}{\sum_{k=1}^n \exp{e_{ik}}}\end{aligned}$$ 

为权重系数（attention score），

$$\begin{aligned} e_{ij}= \frac{(x_i W^Q)(x_j W^K + \gamma_{ij}^K)^T}{\sqrt{d_z}} \end{aligned}$$

是两个输入$$x_i, x_j$$的缩放点集，即缩放后的相关性。

论文把输入序列建模成一个有向全连接图，每一个 token 都是图中的一点。两个 token 之间的边的权重是两个 token 的相对位置编码。


$$\gamma_{ij}^K = w_{\text{clip}}^K(j-i, k)$$

其中 $$\text{clip}(a, b)=\max(\min(a, b), -b)$$ 表示将 $$a$$ 限制在 $$[-b, b]$$ 之间，也即是 $$\gamma_{ij}^K$$ 的取值范围被限制在 $$(w_{-k}^K,..., w_{k}^K)$$ 之间。作者的假设是，token 只需要知道一定范围内的相对位置编码即可，过长是没有必要的。因此，模型只需学习 $$2k+1$$ 个相对位置编码。


前面也提到了相对位置的参数中多头共享的，这样做可以减少参数量，参数空间从 $$hn^2d$$ 可以减少到 $$n^2d$$，其中 $$h$$ 是 head 数量，$$n$$ 是序列长度，$$d$$ 是 embedding 维度。


# FAQ
1. Transformer 如何区分出来 PE (positional encoding) 与 WE (word embedding) 的，为什么不使用 concatenate 而是 sum 呢？

A: PE 向量在前面一些维度上数值变化很大，但是在后面的维度上数值接近于非 0 即 1（见下图-[source](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)），一种[解释](https://github.com/tensorflow/tensor2tensor/issues/1591)是 transformer 在学习过程中，WE 在前面这部分维度上没有编码重要信息,e.g., 词的语义，从这个角度上来看，这跟 concatenate 没有本质区别，但维度更小。 

<img src='https://d33wubrfki0l68.cloudfront.net/ef81ee3018af6ab6f23769031f8961afcdd67c68/3358f/img/transformer_architecture_positional_encoding/positional_encoding.png' width=678pt>


另一种[解释](https://www.reddit.com/r/MachineLearning/comments/cttefo/comment/exs7d08/)是，在高维空间中，PE, WE 是近似正交的。这种情况下，模型可以学习到 PE, WE 的线性组合，而不是简单的 concatenate。上一种解释是这种解释的特例，PE，WE 分别在部分维度上的数值接近于 0，1，这种情况下，PE, WE 可以是近似正交的。


参考: 
- [reddit,Positional Encoding in Transformer](https://www.reddit.com/r/MachineLearning/comments/cttefo/comment/exs7d08/)
- [tensorflow github issue, Why add positional embedding instead of concatenate](https://github.com/tensorflow/tensor2tensor/issues/1591)


2. BERT 是自学习的吗？直接写死效果效果如何？
BERT 使用的是可学习的 positional embedding。

写死的外推性不好。例如，每个位置都按 0, 1, 2, ..., 进行编码，如果序列长度非常长，那么 PE 在数值上占导地位，跟 WE 合并之后，对模型有一定干扰。 

3. PE 跟 positional embedding 区别？
- 构造上，PE 是硬编码，positional embedding 是可学习的参数。


4. 还有哪些位置编码的方法？
- [旋转位置编码 RoPE](https://kexue.fm/archives/8265)
- [More](https://kexue.fm/archives/8130)



# 参考 

- https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1
- [Visual Guide to Transformer Neural Networks - (Part 1) Position Embeddings](https://www.youtube.com/watch?v=dichIcUZfOw)
- [What is positional encoding in the transformer](https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model)
- [Transformer改进之相对位置编码(RPE)-公众号-算法人生](https://mp.weixin.qq.com/s/NPM3w7sIYVLuMYxQ_R6PrA)
- [Transformer 中的位置编码](https://0809zheng.github.io/2022/07/01/posencode.html)


