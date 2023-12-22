---
layout: post
title: Prefix tuning
date: 2023-11-19
tags: prefix-tuning peft
categories: nlp
author: berrysleaf
---
* content
{:toc}


<!-- excerpt -->
Prefix-tuning 是一个轻量级的自然语言生成任务(natural language generation, NLG)的微调方法，它可以在**不改变底层模型参数的情况下，通过修改输入的前缀来优化模型的效果**，在低数据场景下也有较好的表现。





# 背景

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



# 前缀微调 Prefix-tuning 

在 prefix tuning 之前的 prompting 工作主要是人工设计离散的模版或者自动化搜索离散的模版。对于人工设计的模版，模型最终的性能对模版的变化特别敏感，加一个词、少一个词或者位置变动都会造成比较大的变化。而对于自动化搜索模版，成本也比较高；同时，以前这种离散化的 token 搜索出来的结果可能并不是最优的。

除此之外，传统的微调范式利用预训练模型去对不同的下游任务进行 fine-tuning，对每个任务都要保存一份微调后的模型权重，一方面微调整个模型耗时长；另一方面也会占很多存储空间。另外，传统的微调范式对于小数据集的效果不好，因为模型的参数量太大，很容易过拟合。

基于上述两点，[Prefix-tuning](https://arxiv.org/abs/2101.00190) 提出固定预训练 LM，为 LM 添加可训练，任务特定的前缀，这样就可以为不同任务保存不同的前缀，微调成本也小；同时，这种 prefix 实际就是连续可微的 Virtual Token（Soft Prompt/Continuous Prompt），相比离散的 token，更好优化，效果更好。

全量 fine-tuning 太过笨重，一个改进方案是 *lightweight fine-tuning*，思路是**冻结大部分预训练参数，通过添加、微调一小部分可训练模块进行训练**，其实在分类任务中，冻结 BERT，只训练分类头就是这种思路。 Lightweight fine-tuning 中的一个方法是 [*Adapter-tuning*](https://proceedings.mlr.press/v97/houlsby19a.html)，在只微调了 2-4% 参数的情况在，在 NLU 及 NLG 任务上都有不俗的表现。Prefix-tuning 在参数量上比 Adapter-tuning 更加轻量，只需要微调 0.1% 的参数，而且在 NLG 任务上有不错的效果。

但 lightweight fine-tuning 带来的麻烦是，需要在效果与参数量上取得一个平衡，确定训练哪些层是一个需要仔细考虑的问题。

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202312/prefix_vs_fine_tuning_1_20231219_0857.png" width=456pt>
    <img src="https://image.ddot.cc/202312/prefix_vs_fine_tuning_2_20231219_0857.png" width=466pt>
    <figcaption style="text-align:center"> Prefix-tuning V.S. Fine-tuning </figcaption>
</figure>

# 背后的直觉
**在不改变语言模型（LM）参数的情况下，通过提供适当的上下文可以引导 LM**。例如，如果我们想让 LM 生成一个词 Obama，我们可以把常见搭配词（比如 Barack）加入到上下文，这样 LM 就会更有可能生成我们想要的词。直觉上，上下文做为一个“向导”，可以影响 $$x$$ 的编码，并且可以通过引导下一个 token 的分布来影响 $$y$$ 的生成。难点在于，给定一个任务，是否存在这样的上下文并不是显而易见的，人类能轻易理解的指令对 LLM 来说可能是难以理解的。研究人员举了一个例子，给定指令 `summarize the following table in one sentence`，GPT-2 与 BART 都没能对齐这个指令。

## 形式化表示 
用 $$p_\theta(y\vert x)$$ 表示一个参数为 $$\theta$$ 的自回归语言模型（以下简称模型），用符号 $$z=[x;y]$$ 表示 $$x$$ 和 $$y$$ 的拼接，在时间 $$t$$ 第 $$j$$ 层的输出为 $$h_t^{(j)}$$，则模型以 $$z_i$$ 及过去的输出 $$h_{<i}^{(j)}$$ 为输入，计算 $$h_i$$：

$$h_i=p_\theta(z_i, h_{<i})$$

## 技术原理
在输入 token 之前构造一段任务相关的 virtual tokens 作为 prefix，然后训练的时候只更新 prefix 部分的参数，而 PLM 中的其他部分参数固定。根据不同的模型结构，需要构造不同的 prefix：
- 自回归模型结构：在句子前面添加一个 prefix，即原来的输入变成了 $$z=[\text{PREFIX}; x;y]$$，合适的 prefix 可以引导模型生成任务相关的结果；
- Encoder-decoder 模型结构：在 encoder 及 decoder 的输入前面分别添加 prefix，即原来的输入变成了 $$z=[\text{PREFIX}; x; \text{PREFIX}'; y]$$，encoder 端增加前缀是为了引导输入部分的编码，decoder 端增加前缀是为了引导后续token的生成。


# 效果 
## Task 1: tabel to text generation

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202312/prefix_tuning_result_20231222_1059.png" width=789pt>
    <figcaption style="text-align:center"> Prefix-tuning 的效果 </figcaption>
</figure>

尽管 prefix-tuning 只更新了 0.1% 的参数，在 E2E, WebNLG, DART 的效果上
1. 比 adapter tuning (0.1%) 的总体效果要好（笔者注：从数值上来看，没有特别大优势，只在 BLEU 数据集差距明显一些）；
2. 跟 full fine-tuning (100%) 及 adapter tuning (3%) 的效果相当；

## Task 2: low data setting 
<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202312/prefix_tuning_low_data_20231222_1117.png" width=789pt>
    <figcaption style="text-align:center"> Prefix-tuning low data 的效果 </figcaption>
</figure>

在训练数据很少，例如只有几百条甚至几十条数据的情况下，prefix-tuning 的效果要好于 full fine-tuning。上图右是 prefix-tuning 与 fine-tuning 在 summarization 及 tabel-to-text 任务上的效果对比，可以看到 prefix-tuning 在数据量很少的情况下，效果要好于 fine-tuning。左侧是一条 table-to-text 任务下，两种方法给出的答案的直观对比。


## Task 3: extrapolation
<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202312/prefix_tuning_extrapolation_20231222_1134.png" width=789pt>
    <figcaption style="text-align:center"> Prefix-tuning extrapolation 的效果 </figcaption>
</figure>

按照不同的主题（topics）把数据集切分开，在一个数据集上训练，另一个作验证，prefix-tuning 的外推性（泛化性）以微弱的优势好于 fine-tuning。


# 适用场景

当存在大量需要独立训练的任务时，prefix-tuning 就显得非常有优势。一个实际的应用场景是在处理用户隐私，为了保护用户隐私，需要将每个用户的数据分隔开，并分别训练个性化模型。因此，每个用户可以被视为一个独立的任务。如果有数百万用户，prefix-tuning 可以适应这种情况并保持模块化，通过添加或删除前缀，轻松地增加或删除用户，而不会发生交叉污染。

# 扩展阅读 
- Prompt-Tuning：深度解读一种新的微调范式, [https://zhuanlan.zhihu.com/p/618871247](https://zhuanlan.zhihu.com/p/618871247)

