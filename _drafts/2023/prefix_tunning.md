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
- 标签词映射（Label Word Verbalizer） ：因为 `[MASK]` 部分我们只对部分词感兴趣，因此需要建立一个映射关系。例如如果`[MASK]`预测的词是“great”，则认为是 positive 类，如果是 “terrible”，则认为是 negative 类。
- 训练 ：根据 Verbalizer，则可以获得指定 label word 的预测概率分布，并采用交叉信息熵进行训练。此时因为只对预训练好的 MLM head 进行微调，所以避免了过拟合问题。



# Prefix tunning 

主流的 NLP 任务都是 pretraining + fine-tuning 的范式，即在预训练模型的基础上，针对特定任务进行微调。这种方法的优点是简单，但在当下模型越来越大的情况下，fine-tuning 的成本也越来越高。另外，fine-tuning 也有一些缺点，例如，模型的泛化能力不强，对于一些小数据集，模型的效果很差。针对这些问题，有一些研究者提出了一些方法，例如，[《Prefix-Tuning: Optimizing Continuous Prompts for Generation》](https://arxiv.org/pdf/2101.00190.pdf) 就是一种新的 fine-tuning 方法，它可以在不改变模型参数的情况下，通过修改输入的前缀来优化模型的效果。这种方法的优点是可以在不改变模型参数的情况下，优化模型的效果，而且可以在小数据集上取得很好的效果。


# Pattern Exploiting Traing(PTE)

1. 对每一个 pattern，一个单独的 PLM 在一个小的训练集 $\mathcal{\tau}$ 上进行训练，得到一个 pattern-specific PLM。
2. 将所有 pattern-specific PLM 进行 ensemble，使用这些模型进行标注，得到一个软件标签数据集 $\mathcal{D}$。
3. 在  $\mathcal{D}$ 上训练一个分类器。 

