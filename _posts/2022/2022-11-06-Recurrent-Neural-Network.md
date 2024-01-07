---
layout: post
title: Recurrent Neural Network
date: 2022-11-06
tags: rnn nlp
categories: nlp
author: gaoangliu
---
* content
{:toc}


RNN 在很早的时候就被提出来，这个很早指 20 世纪 80、90 年代，虽然当时基本结构已经定型，但由于梯度消失及梯度爆炸的问题，训练非常困难，应用也因此受限。后来 LSTM 及双向 RNN 在同一年（1997）被提出，这两种模型大大改进了早期 RNN 的结构，拓宽了 RNN 的应用范围，为后续发展奠定了基础。后面又有一系列的演变，比如 RNN 语言模型、带有注意力机制的 RNN、ELMo（参照论文阅读笔记 [《ELMo》]({{site.baseurl}}/2023/11/06/ELMo/)） 等等。时至今日，在 transformer 及当下热闹的 LLM 冲击下，RNN 的地位已不复往昔，但作为一个经典的模型，RNN 的结构及其作用还是值得了解的。




RNN 引入了循环连接的思想，允许网络保持一个内部状态，以便在每个时间步捕捉输入之间的依赖关系。这使得 RNN 能够更好地处理变长序列数据，并且能够在输入和输出之间建立序列上的依赖关系。RNN 之所以被称为循环的，是因为它*对序列的每个元素执行相同的任务*，输出依赖于先前的输入。也因此，可以将将 RNN 理解成带有一定“记忆”的网络，能够保存一部分历史信息。


RNN 有多种结构，比如 1 to 1, 1 to n, n to 1, n to n, n to m 等等，这里我们以 n to n 为例，介绍 RNN 的结构及其计算过程。

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/n_to_n_rnn_rc.png" width=578>
    <figcaption style="text-align:center"> 图1. n to n RNN 结构 </figcaption>
</figure>

这种情况下，输入输出长度相同，对应 feed forward 公式：

$$\begin{aligned}
a^{t} &= b + Wh^{(t-1)} + Ux^{t} \\\
h^{t} &= \tanh(a^{t})\\\
o^{t} &= c + Vh^{t} \\\
\hat{y}^{t} &= \text{softmax}(o^{t})
\end{aligned}$$

其中 $$b,c$$ 是偏置项，$$W,U,V$$ 是权重矩阵，$$h^{t}$$ 是隐藏层状态，$$\hat{y}^{t}$$ 是输出层状态，$$x^{t}$$ 是输入层状态。在给定 $$x=(x^{(1)}, ..., x^{(T_x)}), y=(y^{(1)}, ..., y^{(T_y)})$$ 的情况下，RNN 的损失函数为：

$$
\mathcal{L} = -\sum_{t} \log p_\text{model} (y^{t} | x^{(1)}, ..., x^{t})
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

## AG_NEWS 分类小实验
- 学习率 0.001, epoch=15, batch_size=1024

AG_news 训练集上的平均长度（单词数）为 43.28，`max_length` 设置大于 50 后效果较差，应该是过拟合了。作为对比，最大长度设定为 44 时，效果比 25、50 都要好。代码实现: [Here](https://github.com/gaoangliu/gaoangliu.github.io/tree/master/codes/2023/rnn_classification)。

|max_length|num_layers|bidirectional|f1-score(sorted)|precision|recall|
|-|-|-|-|-|-|
|44|1|True|0.8831|0.8833|0.8832|
|44|2|False|0.8818|0.8827|0.8821|
|44|3|False|0.8806|0.8823|0.8806|
|44|3|True|0.8794|0.8804|0.8795|
|44|2|True|0.8775|0.8783|0.8774|
|44|1|False|0.8771|0.8797|0.877|
|50|1|True|0.8766|0.8797|0.8764|
|25|2|True|0.8753|0.8763|0.8753|
|25|3|False|0.8738|0.8746|0.8737|
|50|1|False|0.8738|0.8763|0.874|
|25|1|False|0.8735|0.8746|0.8735|
|25|3|True|0.8724|0.8742|0.8724|
|25|1|True|0.8723|0.8729|0.8723|
|25|2|False|0.8712|0.8722|0.8716|
|50|2|True|0.7914|0.8273|0.8024|
|50|2|False|0.7207|0.7359|0.727|
|50|3|False|0.6566|0.662|0.6927|
|50|3|True|0.5599|0.5999|0.5917|
|60|1|True|0.279|0.4124|0.3616|
|60|1|False|0.2605|0.4047|0.3203|
|60|3|True|0.2363|0.2978|0.2955|
|60|3|False|0.2346|0.2493|0.2734|
|60|2|True|0.214|0.3576|0.2814|
|60|2|False|0.1666|0.3064|0.2633|


# LSTM
RNN 的一个主要问题是，它们很难捕获长期依赖关系。在训练期间，梯度会随着时间的推移指数级地衰减，这意味着网络无法学习与先前输入之间的长期依赖关系。LSTM（Long Short-Term Memory）是一种特殊的 RNN，它通过引入一个称为“单元状态”的新状态来解决这个问题。

相比 RNN 只有一个传递状态 $$h^t$$，LSTM 有两个传递状态 $$h^t, c^t$$，其中 $$c^t$$ 是单元状态（cell state，也称记忆状态），即通常说的长期记忆，$$h^t$$ 是隐藏状态，对应短期记忆。$$c^t$$ 保留历史信息，每次更新会删除一些旧信息，并补充一些新信息，改变相对比较慢，而不同节点下的 $$h^t$$ 会有很大不同。

LSTM 公式比较复杂，但核心只有两点：
1. 隐藏状态 $$h^t$$ 会根据单元状态 $$c^t$$ 来更新； 
2. 单元状态 $$c^t$$ 会根据旧的单元状态 $$c^{t-1}$$, $$h^{t-1}, x^t$$ 来更新。

其它的输入门、遗忘门、输出门这些门控机制等等都是为了实现这两点。

## 更新短期记忆
首先看看 $$h^t$$ 的是如何更新的。回顾在 RNN 中，$$h^t$$ 的更新公式为：

$$h^{t} = \tanh(a^{t})$$

其中 $$a^{t} = b + Wh^{(t-1)} + Ux^{t}$$，$$b$$ 是偏置项，$$W,U$$ 是权重矩阵，$$h^{t}$$ 是隐藏层状态，$$x^{t}$$ 是输入层状态。$$a^{t}$$ 表示。$$a^{t}$$ 这个变换做的事情是**将当前时间步的输入和上一个时间步的隐藏状态通过权重矩阵进行组合，并加上偏置项，以生成当前时间步的隐藏状态输入**。

在 LSTM 中，$$h^t$$ 的更新公式为：

$$h^{t} = o^{t} \odot \tanh(c^{t})$$

其中 $$o^{t}$$ 是输出门，$$\odot$$ 表示逐元素相乘。面输出门的值又依赖于 $$h^{t-1}, x^t$$

$$o^{t} = \sigma(W_{o}[h^{t-1}, x^t] + b_{o})$$

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/lstm_out_20231110.png" width=478>
    <figcaption style="text-align:center"> LSTM 隐藏状态更新 </figcaption>
</figure>

注： $$W_{o}[h^{t-1}, x^t] + b_{o}$$ 的写法跟上面 RNN 中 $$Wh^{(t-1)} + Ux^{t} + b$$ 的写法本质是一样的，LSTM 写法中的 $$W$$ 实际上是

$$W = \begin{bmatrix} W_{h}, 0 \\ 0, U_{x} \end{bmatrix}$$

## 更新长期记忆
单元状态 $$c^t$$ 的更新会考虑两个因素，第一个是上一个时间步的单元状态 $$c^{t-1}$$，第二个是当前时间步的输入 $$x^t$$。
1. 首先，需要决定哪些信息应该被丢弃，这是通过一个称为“遗忘门”的 sigmoid 层（$$f^t$$）来完成的。
2. 然后，需要决定哪些新信息应该被添加到单元状态中，这是通过一个称为“输入门”的 sigmoid 层（$$i^t$$）来完成的。

公式为：

$$c^{t} = f^{t} \odot c^{t-1} + i^{t} \odot \tilde{c}^{t}$$

其中 $$f^{t}$$ 是遗忘门，$$i^{t}$$ 是输入门，$$\tilde{c}^{t}$$ 是新信息，$$\odot$$ 表示逐元素相乘。

遗忘门的值又依赖于 $$h^{t-1}, x^t$$，对应的公式为：

$$f^{t} = \sigma(W_{f}[h^{t-1}, x^t] + b_{f})$$

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/lstm_forget_door_20231110_0716.png" width=478>
    <figcaption style="text-align:center"> 遗忘门 </figcaption>
</figure>

同样输入门 $$i^t$$ 及新的单元状态 $$\tilde{c}^{t}$$的值也依赖于 $$h^{t-1}, x^t$$，对应的公式为：

$$\begin{aligned}
i^{t} &= \sigma(W_{i}[h^{t-1}, x^t] + b_{i}) \\\
\tilde{c}^{t} &= \tanh(W_{c}[h^{t-1}, x^t] + b_{c})
\end{aligned}$$

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/lstm_new_cell_20231110_0736.png" width=478>
    <figcaption style="text-align:center"> 输入门 </figcaption>
</figure>

## 小结 
虽然 LSTM 的公式里面又是输入门、遗忘门，又是输出门的，但各个门的套路（机制）是类似的，都是通过 sigmoid 层来控制信息的流入流出，结构都是 $$z = \sigma(W_z[h^{t-1},x^t] + b_z)$$，这也是 LSTM 被称为门控机制的原因。

三门的作用总结来说：
1. 输入门：控制**新输入数据对记忆细胞的影响**。它通过对输入数据进行加权处理，决定哪些信息可以进入记忆细胞。
2. 遗忘门：控制**记忆细胞中哪些信息需要被遗忘**。通过对先前的记忆状态进行加权处理，学习选择性地遗忘或保留记忆细胞中的信息，防止旧信息长时间积累而不被遗忘。
3. 输出门：控制从**记忆细胞到输出的信息流动**。它通过对记忆细胞状态进行加权处理，控制输出哪些信息给下一层网络或最终输出。

# GRU
GRU 简化了 LSTM 的结构，把输入门和遗忘门合并为一个更新门，同时将隐藏状态和记忆单元合并为一个状态单元。相比于 LSTM，GRU 的参数更少，计算代价也更低，并且很多场景下，效果还差不多。 

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/lstm_vs_gru_20231110_1009.png" width=678>
    <figcaption style="text-align:center"> LSTM v.s. GRU </figcaption>
</figure>

GRU 的更新表达式：

$$h^{t} = (1 - z^{t}) \odot h^{t-1} + z^{t} \odot \tilde{h}^{t}$$

其中 $$z^{t}$$ 是更新门，取值函数 0-1。值越接近于 1，表示“记忆”下来的数据越多，相对地，$$1-z$$ 表示遗忘的信息越多。功能相当于把  LSTM 中的输入门和遗忘门合二为一。 

$$z^{t} = \sigma(W_{z}[h^{t-1}, x^t])$$

$$\tilde{h}^{t}$$ 是新的隐藏状态，相当于 LSTM 中的 $$\tilde{c}^{t}$$，它的计算公式为：

$$\tilde{h}^{t} = \tanh(W_{h}[r^{t} \odot h^{t-1}, x^t])$$

其中 $$r^{t}$$ 是重置门，负责在输入前有选择性地重置记忆细胞的动作，以便适应当前时刻输入的信息。$$r^{t}$$ 的计算公式为：

$$r^{t} = \sigma(W_{r}[h^{t-1}, x^t])$$






# References
- [Recurrent Neural Networks cheatsheet](https://towardsdatascience.com/recurrent-neural-networks-rnns-3f06d7653a85)
- [Deeplearningbook, Chapter 10, Sequence Modeling: Recurrentand Recursive Nets](https://www.deeplearningbook.org/contents/rnn.html)
