---
layout:     post
title:      Tree of Thoughts
date:       2023-12-10
tags:   [prompt, tot]
categories: 
- nlp
---

- [ ] 什么是 ToT？
- [ ] ToT 与 CoT-SC 的区别？
- [ ] 有什么效果？

由 Princeton 及 Google DeepMind 的研究人员提出的一种新的语言模型推理框架“思想树”（Tree of Thoughts, ToT），是 Tree of Chain 的一般形式。ToT 通过将语言模型的推理过程建模为一棵树，从而使得模型能够在推理过程中保持对多个可能性的跟踪，从而提高了模型的推理能力。
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)


<figure style="text-align: center">
    <img src="https://image.ddot.cc/202312/various_approaches_20231212_1609.png" width=789pt>
    <figcaption>图1：ToT 与其他语言模型推理框架的比较</figcaption>
</figure>

从上面的图1可以看出，ToT 实际上在 CoT-SC 的基础上进行了扩展。

CoT 是为解决输入 $x$ 到输出 $y$ 的映射非平凡（non-trivial）的情况，例如 $x$ 是一个数学问题。背后的思想是从 $x$ 到生成 $y$ 的过程中，引入一系列 thoughts $z_i$ 来帮助模型进行推理。对数学题来说，$z_i$ 可以是解决问题的中间步骤。 按本作的形式式符号形式，CoT: $y \sim p_\theta^\text{CoT}(x, z_{1,...,n})$，其中:
- $z_i \sim p_\theta^\text{CoT}(x, z_{1,...,{i-1}})$，$z_1, ..., z_n$ 顺序生成；
- $p_\theta$ 表示参数为 $\theta$ 的预训练语言模型，$p_\theta(x) = \prod_{i=1}^n p_\theta(x_i \vert x_{<i})$。 

CoT-SC（Self-consistency with CoT）是一种集成多个 i.i.d 的 CoT 的方法 $[z_{1, ...,n}^{(i)}, y^{(i)}] \sim p_\theta^\text{CoT}(z_{1,...,n}, y \vert x)$，最后返回出现频率最高的结果 $y=\argmax_y \sum_i \mathbb{I}(y^{(i)}=y)$。

同一个问题通常存在不同的思考过程，例如，定理的证明方式可以有多种不同方式。通过探索更丰富的思考集，输出决策可能更加可靠，这种方式适用于输出空间有限的情况，比如多选问答。 

# 效果


