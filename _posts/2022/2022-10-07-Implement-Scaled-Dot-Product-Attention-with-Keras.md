---
layout: post
title: Implement Scaled Dot-Product Attention with Keras
date: 2022-10-07
tags: nlp transformer
categories: nlp
author: gaoangliu
---
* content
{:toc}



## 回顾一下 Transformer 的结构 
Transformer 是一个 Encoder-Decoder 结构，Encoder（下图 I 左侧所示）的任务是将输入序列转移成连续表示的序列， Decoder（下图 I 右侧所示）接收编码器的输出和前一个时间步骤的解码器输出，以生成一个输出序列。




<img src="https://s3.bmp.ovh/imgs/2022/10/06/2052026273c7533a.png" width=389pt> 

图 I: Transformer 架构 


Transformer 的解码器部分在结构上与编码器有许多相似之处。编码器和解码器在其多头注意力块中都有一个核心部件，那就是缩放点积注意力（Scaled Dot-Product Attention，以下简称 SDPA）。如下图 II 左所示。 

<img src="https://img2022.cnblogs.com/blog/2996113/202210/2996113-20221006175353648-1699935632.png" width=389pt> 

图 II: 缩放点集注意力及多头注意力 

## 缩放点积注意力 

注意力功能可以被描述为一个查询 $$q$$ 和键值对 $$(k,v)$$ 到一个输出的映射，其中查询、键、值和输出全部是向量。输出是值的加权和，其中分配给每个值的权重是根据查询和对应键的兼容性函数计算的。

假设有 $$p$$ 个查询，SDPA 的输入为 $$Q_{d_p \times d_k}, K_{d_p \times d_k}, V_{d_p \times d_v}$$，首先，它会计算 $$Q$$ 与 $$K$$ 的点集 $$P_{d_p \times d_p}$$，然后进行缩放 ($$QK^T/ \sqrt(d_k)$$)，之后把结果经过 softmax 函数，得到一个注意力权重矩阵 $$P'_{d_p \times d_p}$$。 最后，将权重矩阵与值 $$V_{d_p \times d_v}$$ 相乘，即获取最终的注意力矩阵 $$\mathcal{A}_{d_p \times d_v}$$。 整体的运算可由以下公式表示：

$$
\begin{aligned}
\textrm{attention}(Q,K,V) = \textrm{softmax}(\frac{Q K^T}{\sqrt(d_k)}) V 
\end{aligned}
$$


> 以 Q&A 任务为例， k 可以是 question，v 是 answer，q 是新来的 question，计算注意力即统计过去经验里 q 和哪些 k 最相似，然后依葫芦画瓢，根据相似 k 对应的 v，合成当前 question 的 answer。

### Padding Mask
在训练过程中，数据往往是以批量(batch)形式进行训练，对于 NLP 任务来说，一个 batch 的各句话的长度可能有出入，通常我们会采用填充 0 的形式将输入长度补齐到同一长度。但用 0 填充的位置的信息是没有实际意义，我们不希望这个位置参与后期的反向传播过程，以避免最后影响模型自身的效果。因此，在训练过程中我们可以将补全的位置 mask 掉，即通过某种方法将注意力集中到有真实意义的 tokens 对应的位置上。 

Padding mask 采用的方法是将这些位置的值设置成比较大的负值，比如 $$-1*9$$，以保证经过 softmax 运算之后这些 tokens 对应的概率为 0，这种操作相当于把补全位置的无用信息给 mask 掉了。 

举个例子，假设我们的向量为 `vec = [2, 0.5, 0.8, 1, 0, 0, 0, 0]`，最后 4 位是填充部分，那么经过 softmax 运算之后的概率分布为 `[0.41, 0.09, 0.12, 0.15, 0.06, 0.06, 0.06, 0.06]`，但实际上最后 4 位没有数据，也不应该对输出结果产生影响。如果我们将其改造成 `vec = [2, 0.5, 0.8, 1, -1e9, -1e9, -1e9, -1e9]`，那么经过 softmax 运算后的概率分布为 : `[0.53, 0.12, 0.16, 0.19, 0, 0, 0, 0]`。

将 padding mask 考虑在内的注意力计算公式为： 
$$
\begin{aligned}
\textrm{attention}(Q,K,V,M) = \textrm{softmax}(\frac{Q K^T}{\sqrt(d_k)} M) V 
\end{aligned}
$$
这里的 $$M$$ 在非填充位置的值为 $$1$$，填充位置 $$-1e^9$$（或者其他较大的负值）。


### Look-Ahead Mask
在 Decoder 端还有一步 mask，目的是为了避免 decoder 在预测时"作弊"，即在预测第 $$t$$ 个 token 时，解码的时候只能够依靠 $$t$$ 时刻及之前的的输出，而不能依赖于 $$t$$ 时刻之后的输出。

## 实现 

```python
#!/usr/bin/env python
import codefast as cf

from tensorflow import matmul, math, cast, float32
from tensorflow import keras as K
import numpy as np


# Implementing the Scaled-Dot Product Attention
class DotProductAttention(K.layers.Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self,
             queries: np.array,
             keys: np.array,
             values: np.array,
             d_k: int,
             mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(
            cast(d_k, float32))

        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        # Computing the weights by a softmax operation
        weights = K.backend.softmax(scores)

        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)


def test():
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    batch_size = 3  # Batch size from the training process

    input_seq_length = 5  # Maximum length of the input sequence

    queries = np.random.random((batch_size, input_seq_length, d_k))
    keys = np.random.random((batch_size, input_seq_length, d_k))
    values = np.random.random((batch_size, input_seq_length, d_v))
    attention = DotProductAttention()
    print(attention(queries, keys, values, d_k))

if __name__ == '__main__':
    test()
    
```

## 参考
- [SO: Why do we use masking for padding in the Transformer's encoder?](https://stats.stackexchange.com/questions/422890/why-do-we-use-masking-for-padding-in-the-transformers-encoder)