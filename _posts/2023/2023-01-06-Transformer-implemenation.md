---
layout: post
title: Transformer implemenation
date: 2023-01-06
tags: nlp transformer
categories: nlp
author: berrysleaf
---
* content
{:toc}



`forward` 流程：
1. 通过 `self.W_q`、`self.W_k`、`self.W_v` 三个线性层映射到 `d_model` 维度； 



2. 通过 `split_heads` 函数，将 `d_model` 维度的向量分割成 `num_heads` 个 `head_dim` 维度的向量；
3. 通过 `scaled_dot_product_attention` 函数，计算 `attention_scores` 和 `attention_output`；
4. 通过 `concat_heads` 函数，将 `num_heads` 个 `head_dim` 维度的向量拼接成 `d_model` 维度的向量。


```python
def forward(self, query, key, value, mask=None):
    Q = self.W_q(query)
    K = self.W_k(key)
    V = self.W_v(value)

    # split into h heads
    Q = self.split_heads(Q)
    K = self.split_heads(K)
    V = self.split_heads(V)
    
    # scaled dot-product attention
    attn_output, attn_output_weights = self.scaled_dot_product_attention(
        Q, K, V, mask)
    concat_attention = self.concat_heads(attn_output)
    return concat_attention
```

`split_heads` 函数：
1. 将 `d_model` 维度的向量分割成 `num_heads` 个 `head_dim` 维度的向量；

```python
def split_heads(self, x):
    # x: (batch_size, seq_len, d_model)
    # return: (batch_size, num_heads, seq_len, head_dim)
    batch_size, seq_len, d_model = x.size()
    return x.view(batch_size, seq_len, self.num_heads,
                    self.head_dim).transpose(1, 2)
```

# PositionWiseFeedForward
参考之前博客中 [《Attention is all you need - reread》]({{site.baseurl}}/2022/09/30/Attention-is-all-you-need-reread/) 中介绍，FFN 包含了两个线性变换及一个非线性函数 ReLU，其计算公式如下：

$$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

> Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality dff = 2048.

`d_ff`（i.e., `dim_feedforward`） 为 FFN 的隐藏层维度，原文中维度为 2048，FFN 在对特征进行扩展后（512 -> 2048），经过 ReLU（学习复杂的非线性特征），再将特征压缩回 512 维。

```python

class PositionWisefeedForward(torch.nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWisefeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.W_1 = torch.nn.Linear(d_model, d_ff)
        self.W_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.W_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.W_2(x)
        return x
```

如果使用 `nn.Sequential`，则可以简化为：

```python
class PositionWisefeedForward(torch.nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWisefeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return self.ffn(x)
```

完整实现参考 [transformer.py](https://github.com/berrysleaf/berrysleaf.github.io/blob/master/codes/2023/transformer.py)



