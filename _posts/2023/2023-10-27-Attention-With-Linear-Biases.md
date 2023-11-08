---
layout: post
title: Attention With Linear Biases
date: 2023-10-27
tags: nlp position-embedding
categories: nlp
author: berrysleaf
---
* content
{:toc}


- [x] 对比 Sinusodial, RoPE, T5 bias, ALiBi 外推性能



- [x] ALiBi 的机制
- [x] ALiBi 具体实现
- [x] 有哪些应用？
- [x] 速度性能上有什么变化？


之前在苏老师的博客里经常看到外推性这个词，一直不太理解，直到看到了[Attention with Linear Bias](https://arxiv.org/pdf/2108.12409.pdf)这篇论文，才终于明白外推是个什么意思。 

这篇工作其实是关于位置编码的的一个工作，位置编码之前在[positional encoding]({{site.baseurl}}/2022/09/30/Positional-Encoding/)中也有聊过，这个工作主要从 **外推性** 的角度来看位置编码，论证了先前工作的不足，如 Sinusodial, RoPE, T5 bias，这几个经典编码方式在外推性上表现都不太好。 而这个工作提出了一个新的位置编码方式 ALiBi，可以在外推性上有很好的表现。

这个好的表现主要体现在语言模型的 perplexity 随着推理序列长度的增加而保持稳定的能力。这也是外推性的定义，模型在推理时，当序列长度超过训练时的最大长度后，模型保持原有性能的能力。模型基本上确定会在一定长度之后出现性能下降的情况，这个长度就是模型的外推长度。

从 PPL 这个外推性结果上来看，ALiBi 这个工作无疑是很有突破的，但从 21 年提出以来，也就 [BLOOM](https://arxiv.org/abs/2211.05100) 有使用过这种编码方式。[一个推测是](https://blog.csdn.net/weixin_36378508/article/details/133128034)，外推性的指标跟 LLM 的评测指标并不完全 match，或者说不是很正相关，毕竟外推性好只能说明对序列长度的变化不那么敏感，但这跟模型本身的效果关系不大。 

> 注：BloombertGPT(50B), Xuan Yuan(176B) 也有使用 ALiBi。

正弦函数包括后续的改进，在长度外推性上不太好，但后续的大模型还在使用，比如 [LLaMA](https://arxiv.org/pdf/2302.13971.pdf) 使用了 RoPE。RoPE 训练长度 512 时，外推可以增加 200，长度 1024 时，外推可以增加 ？？？。


- 正弦函数（512,1024）外推额外 100 个 token 后，PPL 都开始有明显增加。 比例上，10% 左右。
- 512 的 RoPE 在外推 200 个 token 后，PPL 开始增加，1024 的一样。 比例上，20% 左右。 
- 512 的 T5 bias 外推 1000 个 token （即总长 1512）后，PPL 开始增加，1024 的模型外推 2000 个 token（即总长 3024）后，PPL 开始增加。 比例上，200% 左右。



# 机制

## ALiBi

$$\text{softmax}(q_i K^T + m  \cdot \left[ - (i-1),...,-2,-1,0\right])$$

相当于在 query 和 key 的内积上加了一个**线性偏置**（linear bias），偏置的值是一个等差数列，公差为 1，首项为 $$-m(i-1)$$，其中 $$m$$ 是一个固定的 head-specific slope。当 head 的数量为 8 时，$$m$$ 的值为 0.125。


原来的注意力矩阵为 $$A$$，叠加了 ALiBi 后为 $$A + B \cdot m$$，其中 $$B$$ 是一个 casual 严格下三角矩阵，满足 

\begin{aligned}
B_{ij} &= \begin{cases}
(j-i), & i \geq j \\
0, & i < j
\end{cases}
\end{aligned}

<img src="https://image.ddot.cc/202311/alibi_image.png" width=789pt>


# 效果
学术界没有一个定论。BLOOM 称 ALiBi 的效果要优于 RoPE 及学习得到的位置编码（见下图左），但 [GLM-130B](https://arxiv.org/pdf/2210.02414.pdf) 持相反观点，在 PE 的消融实验中，RoPE 的 PPL 低于 ALiBi（见下图右）。
[《Giraffe: Adventures in expanding context lengths in llms》](https://arxiv.org/pdf/2308.10882.pdf) 中称 ALiBi 表示能力弱，且在 MMLU 等基准测试上效果要比 RoPE 差。

<style>
    figure {
        display: inline-block;
    }
</style>

<figure>
    <img src="https://image.ddot.cc/202311/bloom_alibi_vs_rope_rc.png" alt="BLOOM" width="578pt">
    <figcaption style="text-align: center;">Random</figcaption>
</figure>
<figure>
    <img src="https://image.ddot.cc/202311/alibi_vs_rope_rc.png" alt="GLM-130B" width="578pt">
    <figcaption style="text-align: center;">Random</figcaption>
</figure>


# 扩展
作者（同样三人）在前一年（2020）的工作[《Shortformer: Better Language Modeling Using Shorter Inputs》](https://arxiv.org/pdf/2012.15832.pdf)中还提到了一个 Early Token Curse 的概念，这个概念指的是序列前面的 token 可参考的历史不多，而长序列中 ETC 的比例就比较低，这也是为什么大家普遍认为使用长序列进行训练与推理得到的模型，在 PPL 上效果更好。 

从下图可以看到，随着序列长度的增加，PPL 也在逐渐降低。 

<img src='https://image.ddot.cc/202311/ppl_wrt_sequence_length_rc.png' width=678pt>


ALiBi 的一个实现参考 [Github issue:How can I apply ALiBi Position Encoding into huggingface model?](https://github.com/ofirpress/attention_with_linear_biases/issues/11)。

