---
layout:     post
title:      Prefix Tunning
date:       2023-11-19
tags:   [prefix-tunning, peft]
categories: 
- nlp
---


# Prompt tunning
- [ ] Prefix-tunning 是什么，解决什么问题？
- [ ] 是如何解决这个问题的？
- [ ] 有什么特点？
- [ ] 适用场景？

参考 [Prompt-Tuning：深度解读一种新的微调范式](https://zhuanlan.zhihu.com/p/618871247)

在进入 LLM 时代之前，NLP 任务的主流范式是 pretraining + fine-tuning，即在预训练模型的基础上，针对特定任务进行微调。这种方法的优点是简单，但在当下模型越来越大的情况下，fine-tuning 的成本也越来越高。另外，fine-tuning 也有一些缺点，例如，模型的泛化能力不强，对于一些小数据集，模型的效果很差。

Prompt Tuning 是一种新的 fine-tuning 方法，它可以在不改变模型参数的情况下，通过修改输入的前缀来优化模型的效果。这种方法的优点是可以在不改变模型参数的情况下，优化模型的效果，而且可以在小数据集上取得很好的效果。它将 fine-tuning 的过程转化为一个 prompt 的生成过程，以情感分类为例，给定一个句子，通过添加一个 prompt，生成一个新的句子，然后将新的句子输入到模型中，得到一个概率，这个概率就是情感分类的结果。这个过程可以看成是一个生成式的过程，即给定一个输入，生成一个输出。这个过程可以用下面的公式表示：

$$\begin{aligned}
\mathcal{L}(\theta) &= \sum_{(x, y)\in \mathcal{D}} \log p_\theta(y\lvert x) \\\
&= \sum_{(x, y)\in \mathcal{D}} \log p_\theta(y\lvert x, \text{prompt}(x))
\end{aligned}$$


Prompt-Tuning 的一般流程：
- 构建模板（Template Construction）：通过人工定义、自动搜索、文本生成等方法，生成与给定句子相关的一个含有 `[MASK]` 标记的模板。例如 `It was [MASK].`，并拼接到原始的文本中，获得Prompt-Tuning 的输入：`[CLS] I like the Disney films very much. [SEP] It was [MASK]. [SEP]`。将其喂入 BERT 模型中，并复用预训练好的 MLM 分类器，即可直接得到`[MASK]`预测的各个 token 的概率分布；
- 标签词映射（Label Word Verbalizer） ：因为 `[MASK]` 部分我们只对部分词感兴趣，因此需要建立一个映射关系。例如如果`[MASK]`预测的词是“great"，则认为是 positive 类，如果是 “terrible"，则认为是 negative 类。
- 训练 ：根据 Verbalizer，则可以获得指定 label word 的预测概率分布，并采用交叉信息熵进行训练。此时因为只对预训练好的 MLM head 进行微调，所以避免了过拟合问题。



# Prefix tunning 

主流的 NLP 任务都是 pretraining + fine-tuning 的范式，即在预训练模型的基础上，针对特定任务进行微调。这种方法的优点是简单，但在当下模型越来越大的情况下，fine-tuning 的成本也越来越高。另外，fine-tuning 也有一些缺点，例如，模型的泛化能力不强，对于一些小数据集，模型的效果很差。针对这些问题，有一些研究者提出了一些方法，例如，[《Prefix-Tuning: Optimizing Continuous Prompts for Generation》](https://arxiv.org/pdf/2101.00190.pdf) 就是一种新的 fine-tuning 方法，它可以在不改变模型参数的情况下，通过修改输入的前缀来优化模型的效果。这种方法的优点是可以在不改变模型参数的情况下，优化模型的效果，而且可以在小数据集上取得很好的效果。


# Pattern Exploiting Training(PTE)
这个工作算是 prompt 范式的开山之作，prompt tunning 的思想其实很早就有了，比如使用 GPT-2 将文本分类任务转换成问答任务（参考论文：[Zero-shot Text Classification With Generative
Language Models](https://arxiv.org/pdf/1912.10165.pdf)）。Prompt 的思想是对**输入进行改造，挖掘语言模型的潜力，获得任务相关的输出，从而避免精调模式带来的灾难性遗忘问题**。因此要考虑的问题是：

1. 如何设计合适的 Prompt，激发模型的潜能。
2. 输出结果如何跟最终的任务结果关联起来。比如用生成模型作情感分类，如何将生成的结果映射到情感值上。 
3. 存在一定标注数据的情况下， 如何微调模型，使得模型能够学到任务相关的知识。

这个工作的贡献是提出了一种通用的方法，可以将不同任务转换成一个**模板填空**的任务，然后使用预训练模型进行微调。这个方法的优点是可以在不改变模型参数的情况下，优化模型的效果，而且可以在小数据集上取得很好的效果。大概流程：

1. 使用少量的标注数据，对每一个 prompt 训练一个语言模型（LM）； 
2. 使用多个 prompt 模型对未标注数据进行伪标注，得到一个软标签数据集；
3. 在软标签数据集上训练一个模型，得到最终的模型。

## 细节

令 $M$ 表示一个语言模型，$V$ 表示词汇表，$\mathcal{L}$ 表示分类任务的标签集合。令 $s_i \in \Omega = V^*$ 表示一条序列，这个序列可以是一条短语或者一条句子。 

1. *pattern*，一个 pattern $P: \Omega^k \rightarrow \Omega$ 是一个将一个序列集合映射到一条序列的函数。输入 $x = (s_1, ..., s_k) \in \Omega^k$ 表示由 $k$ 个序列构成的集合。例如，当 $k=2$ 时，$x=(s,t)$ 由两条序列构成，对应的任务 $T$ 可以是一个相似度判断任务。 
2. *verbalizer*， $v:\mathcal{L} \rightarrow V$，将每一个标签映射到词汇表中的一个词。例如，对于情感分类任务，可以定义两个 verbalizer，分别是 $v_+$ 和 $v_-$，将 positive 和 negative 映射到词汇表中的一个词。

Pattern 是对原始输入进行格式转换，适配下游模型，Verbalizer 是将下游模型的输出映射到任务标签上。以 Yelp 评论打分为例，给定一条评论 $r$，可能的 patterns 有：
1. $P_1(r) = \text{It was [M]. } r$; 
2. $P_1(r) = r \text{. All in all, it was [M]. }$; 

其中 $\text{[M]} \in V$ 表示一个 mask 词。对应的 verbalizer 为 $v: [1,2,3,4,5] \rightarrow \{ \text{great, good, okay, bad, terrible} \}$.

## 训练 
Pattern 实际上把输入序列转成 MLM 的输入，MLM 对这个序列进行预测，得到一个概率分布，用 $M(w | z), w \in V, z \in \Omega$。 给定  $p=(P, v)$，定义 $l \in \mathcal{L}$ 的得分为：

$$s_p(l) = M(v(l) \lvert P(x))$$

通过 softmax 函数，可以得到一个概率分布：

$$q_p(l) = \frac{\exp(s_p(l))}{\sum_{l' \in \mathcal{L}} \exp(s_p(l'))}$$


## PVP 集成 
PVP 的选择对效果有直接的影响，但有监督数据 $\mathcal{T}$ 比较少，不可能有验证集来验证一个 PVP 的效果。 一个方法是根据直觉定义一批 PVPs $\mathcal{P}$，对每一个  $p\in \mathcal{P}$ 都微调一个 $M_p$。然后集成这样模型 $\mathcal{M} = \{ M_p | p \in \mathcal{P} \}$ 对无标注样本集 $\mathcal{D}$ 进行伪标注:

$$s_\mathcal{M}(l |x) = \frac{1}{Z} \sum_{p \in \mathcal{P}} w(p) \cdot s_p(l | x) $$

其中，$Z= \sum w(p)$，$w(p)$ 表示每一对 $(P, v)$ 的权重。权重的设计有很多种方案，简直暴力一点的，可以将所有权重都调成相等的值，即 $w(p) = 1$，或者引入先验知识赋予不同的值。 文中给定的另一种方案是 $w(p) = \text{precision}_{p, \mathcal{T}}$，即使用 $p$ 在训练集 $\mathcal{T}$ 的精度。在论文实验（Table 4）中，两种权重方案的效果没有明显差异 ($\lvert \mathcal{T} \rvert = 10$)。 

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202311/pet_20231129_1423.png" width=678pt>
    <figcaption style="text-align: center;"> PET schematic representation </figcaption>
</figure>

## iPET

迭代版本的 PET，通过**训练——模型标注——再训练——再标注**的迭代 $k$ 次获得最终数据集。 

iPET 优势是什么？优势是标注的更准确。作者设计了一个实验， 在 Yelp、AG News 等 4 个数据集上使用 iPET 进行 zero-shot 学习，总迭代轮次等于 4。对 AG News、Yahoo 任务还设计了跳过第二、三次迭代，即第一次迭代后直接伪标注 $d^3 \cdot \lvert \mathcal{T}_1 \rvert$ 个数据供 $\mathcal{M}_4$ 训练。 实验有如下结果：

1. 随着迭代次数增加，模型效果也逐渐提升； 
2. 跳过中间迭代过程，一次性伪标注相同数量样本并训练的效果**弱于逐步迭代**的效果。 


在每一步迭代时，只选择模型比较自信的样本做为下一步的训练数据，这样可以相较于一步到位的标注，错标的数据更少。 通过实验验证，当数据集大小较小时，比如只有几十个样本时，iPET 能带来几个点的效果提升。

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/ipet_generation_result_20231201_1401.png" width=678pt>
    <figcaption style="text-align: center;"> iPET schematic representation </figcaption>
</figure>


几个点：
1. 每次标注时，不是选择所有模型，而是选择模型的一个子集进行标注； 
2. 在每一次迭代中，数据等比缩放。 通过随机采样，保证新标注数据中各标签的比例都与原始数据中的比例保持一致，即 $\lvert \mathcal{T}_i \rvert = d \cdot \lvert \mathcal{T}_{i-1} \rvert $，其中 $d$ 是缩放比例。在原文实验中，$d$ 设置为 5，迭代终止条件是每个模型都最终在 1000 条样本上训练，即轮次 $k = \lceil \log_d(1000 / \lvert \mathcal{T} \rvert)$；
3. 在整个未标注数据集 $\mathcal{D}$ 上都进行伪标注，得到 $\mathcal{T}_\mathcal{N} = \{ (x, \argmax_{\substack{l \in \mathcal{L}}} s_\mathcal{N}(l | x)) | x \in \mathcal{D}\}$，在下一步训练使用时，从所有伪标数据集中抽取一部分。一对样本 $(x, y)$ 被抽取的概率正比于 $s_\mathcal{N}(l|x)$。

## 效果 
### 纵向对比  
PET 对比 supervised 方案。 整体上而言，训练集 $\mathcal{T}$ 越小，PET 提升越大。以分类任务 Yelp, AG News 例，iPET 在 $\lvert \mathcal{T} \rvert = 10$ 上的提升分别为 0.365 和 0.642，而在 $\lvert \mathcal{T} \rvert = 100$ 上的提升分别为 0.099 和 0.036。因此一个粗糙的结论：**对于分类任务，如果在几百条标数据上效果不好，那个效果基本上也接近半监督学习的上限了**。
差异比较大的是 MNLI 数据集，在 $\lvert \mathcal{T} \rvert = 1000$ 时，iPET 对比 supervised 仍然有 10 个点的提升。

### 横向对比
对比两个基于数据增强的方案 UDA、MixText， PET 在数据量比较小（小于 50）时都表现出明显的优势。  



<figure style="text-align:center">
    <img src="https://image.ddot.cc/202311/ipet_vs_supervised.png" width=678pt>
    <figcaption style="text-align: center;"> iPET v.s. Supervised </figcaption>
</figure>



## QA
1. 如何利用 MLM 的？
2. 为什么这种方法会有效果？
3. 跟主动学习有什么区别？ 
4. 为什么不直接用伪标签？






