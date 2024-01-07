---
layout: post
title: Transformer Encoder
date: 2022-10-06
tags: nlp transformer encoder
categories: nlp
author: gaonagliu
---
* content
{:toc}



## 回顾一下 Transformer 的结构 

Transformer 是一个 Encoder-Decoder 结构，Encoder（下图 I 左侧所示）的任务是将输入序



列转移成连续表示的序列， Decode（下图 I 右侧所示）接收编码器的输出和前一个时间步骤的解码器输出，以生成一个输出序列。在生成输出序列过程中， Transformer 不依赖于卷积及递补。 

<center> <img src="https://s3.bmp.ovh/imgs/2022/10/06/2052026273c7533a.png" width=389pt> </center>
<center> 图 I: Transformer 架构 </center> 

> 注：关于 transformer 原文的阅读笔记可参考[Attention is all you need - reread
]({{site.baseurl}}/2022/09/30/Attention-is-all-you-need-reread/)

## Encoder 
Encoder 由一叠（原文中有 6 层）相同的 encoder 层构成，每个 encoder 层又主要包含两个子层：
- 第一个子层包括一个多头注意力机制，接收 queries, keys 及 values 作为输入
- 第二层包含一个全连接前馈网络 
这两个子层之后都接一个 normalization 层，这一层通过残差连接同时接收上一层的输入与输出做为它的输入，即输入形式为：`LayerNorm(Sublayer Input + Sublayer Output)`

查询、键和值带有相同的输入序列，这些序列被嵌入并增加了位置信息，其中查询和键的维度为 $$d_k$$，而值的维度为 $$d_v$$。

Vaswani 等人在实现中在每一个子层与 norm 层之间添加了 dropout，为模型引入了正则。 

## 使用 Keras 进行实现
先从全连接层开始实现。 全连接前馈网络由两个线性变换组成，中间有一个 ReLU 激活。第一个线性变换产生一个维度为 2048 的输出，而第二个线性变换产生一个维度为 512 的输出。

1. 通过继承 `keras.layers.Layer` 类，然后添加一个方法 `call()`, 接收一个输入，并通过两个全连接层与 ReLU 激活，返回一个维度为 512 的输出。 

```python
import tensorflow as tf
from tensorflow import keras as K

class FeedForward(K.layers.Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.f1 = K.layers.Dense(d_ff)  # First fully connected layer
        self.f2 = K.layers.Dense(d_model)  # Second fully connected layer
        self.activation = K.layers.ReLU()  # ReLU activation layer

    def call(self, x):
        # The input is passed into the two fully-connected layers, with a ReLU in between
        x_fc1 = self.f1(x)
        return self.f2(self.activation(x_fc1))
```

2. 创建 `AddNormalization` 层，通过残差连接接收、组合并规范化 encoder 每一个子层的输入与输出。 
```python
class AddNormalization(K.layers.Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = K.layers.LayerNormalization(
        )  # Layer normalization layer

    def call(self, x, sublayer_x):
        # The sublayer input and output need to be of the same shape to be summed
        add = x + sublayer_x
        # Apply layer normalization to the sum
        return self.layer_norm(add)
```

<center> <img src="http://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png" width=389pt> </center>
<center> 图 II：encoder 内部结构 </center> 


## 参考
- [Implenmeting the Transformer Encoder from Scratch in TensorFlow and Keras](https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras/)
