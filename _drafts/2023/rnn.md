---
layout:     post
title:      Recurrent Neural Network
date:       2022-11-06
tags:   [rnn, nlp]
categories: 
- nlp
---

RNN 在很早的时候就被提出来，这个很早指 20 世纪 80、90 年代，虽然当时基本结构已经定型，但由于梯度消失及梯度爆炸的问题，训练非常困难，应用也因此受限。后来 LSTM 及双向 RNN 在同一年（1997）被提出，这两种模型大大改进了早期 RNN 的结构，拓宽了 RNN 的应用范围，为后续发展奠定了基础。后面又有一系列的演变，比如 RNN 语言模型、带有注意力机制的 RNN、ELMo（参照论文阅读笔记 [《ELMo》]({{site.baseurl}}/2023/11/06/ELMo/)） 等等。时至今日，在 transformer 及当下热闹的 LLM 冲击下，RNN 的地位已不复往昔，但作为一个经典的模型，RNN 的结构及其作用还是值得了解的。

RNN 引入了循环连接的思想，允许网络保持一个内部状态，以便在每个时间步捕捉输入之间的依赖关系。这使得 RNN 能够更好地处理变长序列数据，并且能够在输入和输出之间建立序列上的依赖关系。RNN 之所以被称为循环的，是因为它*对序列的每个元素执行相同的任务*，输出依赖于先前的输入。也因此，可以将将 RNN 理解成带有一定“记忆”的网络，能够保存一部分历史信息。


RNN 有多种结构，比如 1 to 1, 1 to n, n to 1, n to n, n to m 等等，这里我们以 n to n 为例，介绍 RNN 的结构及其计算过程。

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/n_to_n_rnn_rc.png" width=578>
    <figcaption style="text-align:center"> 图1. n to n RNN 结构 </figcaption>
</figure>

这种情况下，输入输出长度相同，对应 feed forward 公式：

$$\begin{aligned}
a^{(t)} &= b + Wh^{(t-1)} + Ux^{(t)} \\\
h^{(t)} &= \tanh(a^{(t)})\\\
o^{(t)} &= c + Vh^{(t)} \\\
\hat{y}^{(t)} &= \text{softmax}(o^{(t)})
\end{aligned}$$

其中 $b,c$ 是偏置项，$W,U,V$ 是权重矩阵，$h^{(t)}$ 是隐藏层状态，$\hat{y}^{(t)}$ 是输出层状态，$x^{(t)}$ 是输入层状态。在给定 $x=(x^{(1)}, ..., x^{(T_x)}), y=(y^{(1)}, ..., y^{(T_y)})$ 的情况下，RNN 的损失函数为：

$$
\mathcal{L} = -\sum_{t} \log p_\text{model} (y^{(t)} | x^{(1)}, ..., x^{(t)})
$$


## RNN 实现 

结构示例
```python
import torch
from torch.nn import RNN

input_size = 12
hidden_size = 20
num_layers = 2
batch_size = 3
sequence_length = 10

rnn = RNN(input_size, hidden_size, num_layers, batch_first=True)
inputs = torch.randn(batch_size, sequence_length, input_size)
h0 = torch.randn(num_layers, batch_size, hidden_size)
output, hn = rnn(inputs, h0)
print(output.shape)
```

需要注意的是当 `batch_first` 的值发生变化时，`inputs, outputs` 的维度也会发生变化，细节参考 [torch 官方文档](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)。


# LSTM
RNN 的一个主要问题是，它们很难捕获长期依赖关系。在训练期间，梯度会随着时间的推移指数级地衰减，这意味着网络无法学习与先前输入之间的长期依赖关系。LSTM（Long Short-Term Memory）是一种特殊的 RNN，它通过引入一个称为“单元状态”的新状态来解决这个问题。LSTM 通过一个称为“门”的结构来控制单元状态的信息流，以便在训练期间保留或丢弃信息。LSTM 包含三个门：输入门、遗忘门和输出门。这些门控制着单元状态的信息流，以便在训练期间保留或丢弃信息。

## 输入门
输入门控制着新信息的流入，它由一个 sigmoid 层和一个 tanh 层组成。sigmoid 层输出一个介于 0 和 1 之间的值，表示应该保留多少新信息。tanh 层输出一个介于 -1 和 1 之间的值，表示新信息的值。sigmoid 层的输出和 tanh 层的输出相乘，得到的结果表示应该保留多少新信息。

# References
- [Recurrent Neural Networks cheatsheet](https://towardsdatascience.com/recurrent-neural-networks-rnns-3f06d7653a85)
- [Deeplearningbook, Chapter 10, Sequence Modeling: Recurrentand Recursive Nets](https://www.deeplearningbook.org/contents/rnn.html)
