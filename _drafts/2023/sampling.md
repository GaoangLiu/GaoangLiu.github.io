---
layout:     post
title:      也说解码策略 
date:       2023-11-11
tags: [nlp, sampling]
categories: 
- nlp
---

对于自回归模型，一个比较重要的问题是如何解码。贪婪解码（greedy decoding）是最简单的解码方法，它在每个时间步选择概率最大的词作为输出。贪婪解码的优点是简单高效，但是它的缺点也很明显，即容易陷入局部最优，导致输出文本不连贯，或者重复循环。为了解决这个问题，人们提出了束搜索（beam search）等解码策略。束搜索通过宽度优先搜索创建搜索树，在每个时间步保留概率最大的 $k$ 个序列，然后在下一个时间步对这 $k$ 个序列的所有可能的扩展序列分别计算概率，再次选择概率最大的 $k$ 个序列，以此类推。束搜索的计算量大，而且束搜索也不能完全避免陷入局部最优的问题。论文[《The Curious Case of Neural Text Degeneration》](https://arxiv.org/pdf/1904.09751.pdf)(ICLR 2020) 指出这种基于最大化的解码方法仍然会导致**退化**现象，即产生苍白（乏善可陈）、不连贯或陷入重复循环的输出文本。

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/beam_search_vs_human_20231111_0828.png" width=478>
    <figcaption style="text-align:center"> 图1. 束搜索与人类的对比 </figcaption>
</figure>


# 形式化描述 
给定由 $m$ 个词元(tokens)构成的上下文序列 $x_1,...,x_m$，解码任务是生成后续 $n$ 个词元$x_{m+1},...,x_{m+n}$ 来补齐这个序列。可能的生成方法有很多种，常用的指导思想找到最优的序列 $x_{m+1},...,x_{m+n}$，使得其概率最大，即：

$$\begin{aligned}
\hat{x}_{m+1},...,\hat{x}_{m+n} &= \arg \max_{x_{m+1},...,x_{m+n}} p(x_{m+1},...,x_{m+n}|x_1,...,x_m) \\\
&= \arg \max_{x_{m+1},...,x_{m+n}} \prod_{i=1}^{n} p(x_{m+i}|x_1,...,x_{m+i-1})
\end{aligned}$$

# 贪婪解码
要获取最大概率的序列，最直观的方法是枚举所有可能的序列，然后选择概率最大的序列。但是这种方法的计算复杂度是指数级的，因此实际中不可行。贪婪解码在每个时间步选择概率最大的词作为输出。优点是简单高效，但容易陷入局部最优，导致输出文本不连贯，或者重复循环。

# 集束搜索

相当于暴力搜索与贪婪解码的折中，对比贪婪解码，集束搜索增大了搜索空间，计算复杂度是 $O(k \cdot n \cdot V)$，其中 $k$ 是束宽，$n$ 是序列长度，$V$ 是词表大小。当 $k=1$ 时，集束搜索退化为贪婪解码。

注： $n$ 选 top $k$ 有[Quickselect](https://en.wikipedia.org/wiki/Quickselect)算法，平均复杂度为 $O(n)$，不需要 $O(n\log n)$ 的排序。QuickSelect 的思路可以参考之前的笔记[《Algorithm》]({{site.baseurl}}/2019/06/23/Algorithm)。


# Top-k 采样（top-k sampling）
Greedy & Beam Search 的思想都是生成概率最大的序列，这种思路生成的句子有两点不足。一是信息量较少，二是缺乏多样性，毕竟人的表述方式总是“不拘一格”的，在一句话中，真人语言所用的词汇很少一直使用高频常见词，而更可能在表述过程中，突然转向使用低频但更富有信息量的词汇，这也是真人语言的一个内在属性。 

针对这个问题的一个解决方案是引入随机性，比如在确定下一个词时，不总是选择概率最大的词，而是从概率最大的 $k$ 个词中随机选择一个，这种方法称为 top-k 采样，操作步骤：

1. 计算 $p(x_{m+1} \vert x_1,...,x_{m})$，选择概率最大的 $k$ 个词，记为 $V^k$；
2. 归一化 $V^k$ 中词的概率，得到新的概率分布 $p'$，$p'(x) = p(x)/\sum_{x \in V^k} p(x)$；
3. 根据 $p$ 中采样一个词 $x_{m+1}$，作为下一个词。

Top-K 适合候选词较多且分布较平滑的场景，而有些情况下，概率可能集中分布在几个候选词上（如下图），如果 $k$ 较小，在某些语境中就有可能产生空洞、泛泛而谈的文本；而如果 $k$ 较大，前 $k$ 词汇表中就会包含不合适的候选词，这些候选词被归一化化的概率就会增加。

这种情况下，$k$ 值不是很好界定，但可以通过设定一个阈值 $\delta$，动态的从候选词中确定一个词表 $V^p$，使得这些词的概率之和不大于 $\delta$，i.e., 

$$\sum_{x \in V^p} p(x_{m+1}|x_1,...,x_{m}) \geq \delta$$

这种方法称为 top-p 采样，也称核采样。

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/peak_distribution_20231111_1330.png" width=678>
    <figcaption style="text-align:center"> Peaked Distribution </figcaption>
</figure>

# 核采样（nucleus sampling）
与 top-k 采样类似，将词表缩小到一定范围，然后对概率分布进行缩放得到一个新的分布 $p'$，从 $p'$ 中采样一个词作为下一个词。两个方案本质上都**舍弃长尾词并重缩放头部词概率分布**的方案。

那这种采样方式真的有用吗？原作作者评估了不同采样策略生成的文本的 perplexity，并表明由 top-p 方案获得的文本在 perplexity 是最接近于人类创作的文本，且论证了 perplexity 并非越低越好，比如 greedy 策略的 perplexity 非常低，但产生的文本都是重复的，多样性也比较低。 

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/topp-perplexity_20231115_0747.png" width=678>
    <figcaption style="text-align:center"> Peaked Distribution </figcaption>
</figure>

除了上述几个方案外，还有一个常用的方案是 temperature sampling，即在确定下一个 token 时，先对原有词表的概率分布进行缩放，然后再从缩放后的分布中采样。缩放公式是：

$$p'_\tau(x) = \frac{\exp(p(x)/\tau)}{\sum_{x \in V} \exp(p(x)/\tau)}$$

其中 $\tau$ 是温度，$\tau \in (0, 1]$，$\tau$ 越大。新的概率分布越接近于均匀分布，模型选择冷门词的概率就增大（如下图 $y(\tau=0.9)$ 的分布），因此有可能导致生成不连贯、不合乎逻辑的序列。$\tau$ 越小，概率分布越不均匀，高频词将分配到更高的概率，模型越倾向于选择出现概率大的词汇（如下图 $y(\tau=0.1)$ 的分布），生成结果也就越确定，对应的内容越保守。 

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202311/temperature_sampling_20231111_1551.png" width=789>
    <figcaption style="text-align:center"> 设置不同温度时概率分布的变化 </figcaption>
</figure>


# 参考 
- [图说文本生成解码策略](https://finisky.github.io/illustrated-decoding-strategies/
)
- [The Curious Case Of Neural Text Degeneration](https://arxiv.org/pdf/1904.09751.pdf)

