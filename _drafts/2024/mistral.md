---
layout:     post
title:      The mysterious Mistral (WIP)
date:       2024-01-11
tags: [mistral, moe]
categories: 
- nlp
---

Mistral.ai 在 arXiv 上放出来论文 [Mixtral of Experts](https://arxiv.org/pdf/2401.04088.pdf)，整体结构上跟 Mixtral 7B 一样，唯一的区别在于每个层都由 8 个前馈块（即专家）组成。







# Mistral 7B
论文: https://arxiv.org/pdf/2310.06825.pdf, arXiv 23.10

首先说一下当时这篇工作的亮点：
1. 在所有基准测试中均优于当时最佳开源模型 Llama 2 13B，在推理、数学和代码生成方面优于 Llama 1 34B。
2. 利用分组查询注意力（GQA）来实现更快的推理，并结合滑动窗口注意力（SWA）来有效地处理任意长度的序列，同时降低推理成本。
3. 提供了一个经过指令微调的模型，Mistral 7B-Instruct，在人类和自动化基准测试上都超越了 Llama 2 13B 聊天模型。


# Sliding Window Attention (SWA) 
在 vanilla attention 中，每个 token 都会与所有其他 token 进行交互，这样的话，计算复杂度就是 $O(n^2)$，其中 $n$ 是序列长度。这样的计算复杂度在序列很长时会很高，因此需要一种更高效的方法。

SWA 最早在 [LongFormer](https://arxiv.org/pdf/2004.05150.pdf)中提出，目标是 LLM 中有效的处理长序列。思想是把每个 token 的注意力跨度限制到它周围的固定窗口中。LongFormer的做法是定义一个宽度为$W$的窗口，使得 query 节点只能注意到对等的 key 节点以及 key nodes 左右 $W/2$ 个节点。这样的话，计算复杂度就是 $O(W \cdot n)$，其中 $n$ 是序列长度。当 $W<<n$，计算复杂度就会大大降低。

这样，其他 key nodes 的信息是不是丢失了？也不是，多个层堆叠在一起时，在高层上，query 节点会间接的注意到远处的 key 节点。假设有 $l$ 层，每层的窗口大小为 $W$，那么在第 $l$ 层，query 节点可以间接的注意到 $W*l$ 范围内的 key 节点。

Mistral 7B 中的 window size 是 4096.

# Grouped Query Attention (GQA)
目标是在推理时减少计算量。

# Byte-fallback BPE algorithm
提高 LLM 的 tokenization，提高收敛速度和模型质量。

<!-- https://huggingface.co/blog/moe -->

# Sparse Mixtures of Experts
给定输入 $x$，MoE 模块的输出 $y=\sum_{i=0}^{n-1} G(x)_i \cdot E_i(x)$，其中 $n$ 是专家网络（下称专家）的个数，$G(x)$ 是第$i$专家的权重，$E_i(x)$ 是第 $i$ 个专家的输出。$G(x)$ 的每个元素都是一个概率值，且满足 $\sum_{i=0}^{n-1} G(x)_i = 1$。

那 SMoE 中的 sparse 是什么意思呢？其实就是指只有少数专家参与决策，即 $G(x)$ 中的大部分元素都是 0，只有少部分非零。这样的话，就可以减少计算量。既然只有少数专家参与决策，那就需要一个策略决定哪些专家参与决策，方法有很多种，一种简单高效的方法是对线性层的前K个logits应用softmax函数：

$$G(x)=\text{Softmax}(\text{TopK}(x \cdot W_g))$$

$k$ 做为一个超参数，可以通过平衡效果与计算量来调整。Mistral 8x7B 中使用的是 $k=2$，即只有两个专家参与决策。

# 直接偏好优化

直接偏好优化（Direct Perference Optimization,DPO）是一种 LM 偏好对齐算法，最初 Rafailov 等人在 [《Direct Preference Optimization: Your Language Model is Secretly a Reward Model》](https://arxiv.org/abs/2305.18290) 中提出。

在此之前，让LM对人类偏好对齐常用的算法是RLHF，思路是先根据人类偏好拟合一个奖励模型，再用强化学习的方法去微调一个LM，使得LM的输出尽可能奖励最大化。但RLHF复杂，且不稳定。对比基于 Proximal Policy Optimization(PPO) 的 RLHF的方案，DPO优点是稳定、效果好、计算量小。


### 📝 一些与决策相关的工作 
- [Unified scaling laws for routed language models](https://arxiv.org/abs/2202.01169)
- [Dselect-k: Differentiable selection in the mixture of experts with applications to multi-task learning](https://proceedings.neurips.cc/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html)
- [CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge](https://arxiv.org/abs/1811.00937)
 
Mistral 的 MoE 层结构如下图所示：

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202401/mistral-moe-layer_20240111_1038.png" width=789pt>
    <figcaption style="text-align:center"> Mistral MoE layer 结构图 </figcaption>
</figure>


# Q & A 环节 
## Router network 是怎么工作的？怎么选择，怎么组合？
从上面的结构图里可以知道每一个 router 是一个 gating network，它的输入是 $x$，输出是 $G(x)$，即每个专家的权重。这个 gating network 做为整个模型的一部分，在训练时参数 $W_g$ 也是通过反向传播来更新的。

一旦训练完成，推理过程中直接使用 $G(x)$ 的值即可。

## 为什么要用 router network？
理论上$n$个专家共同决策，效果更稳定一些（泛化更好），但要平衡计算量和效果，所以只选择了部分专家。Router network 就是这样的一个策略，它决定哪些专家参与决策。

## 为什么要用两个expert？