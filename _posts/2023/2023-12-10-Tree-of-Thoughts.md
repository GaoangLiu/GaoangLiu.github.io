---
layout: post
title: Tree of Thoughts
date: 2023-12-10
tags: prompt tot
categories: nlp
author: berrysleaf
---
* content
{:toc}


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

CoT 是为解决输入 $$x$$ 到输出 $$y$$ 的映射非平凡（non-trivial）的情况，例如 $$x$$ 是一个数学问题。背后的思想是从 $$x$$ 到生成 $$y$$ 的过程中，引入一系列 thoughts $$z_i$$ 来帮助模型进行推理。对数学题来说，$$z_i$$ 可以是解决问题的中间步骤。 按本作的形式式符号形式，CoT: $$y \sim p_\theta^\text{CoT}(x, z_{1,...,n})$$，其中:
- $$z_i \sim p_\theta^\text{CoT}(x, z_{1,...,{i-1}})$$，$$z_1, ..., z_n$$ 顺序生成；
- $$p_\theta$$ 表示参数为 $$\theta$$ 的预训练语言模型，$$p_\theta(x) = \prod_{i=1}^n p_\theta(x_i \vert x_{<i})$$。 

CoT-SC（Self-consistency with CoT）是一种集成多个 i.i.d 的 CoT 的方法 $$[z_{1, ...,n}^{(i)}, y^{(i)}] \sim p_\theta^\text{CoT}(z_{1,...,n}, y \vert x)$$，最后返回出现频率最高的结果 $$y=\argmax_y \sum_i \mathbb{I}(y^{(i)}=y)$$。

同一个问题通常存在不同的思考过程，例如，定理的证明方式可以有多种不同方式。通过探索更丰富的思考集，输出决策可能更加可靠，这种方式适用于输出空间有限的情况，比如多选问答。 

人类在解决问题会采用的一种思路是，将问题拆成若干个过程，每一个过程都有多个解决方案，这些过程与方案构成一个树状结构，人在这棵树上进行搜索，逐渐找到最终的解决路径。 选择哪个分支是由启发式决定的，这些启发式有助于确定问题空间并引导问题解决者找到解决方案。

ToT 的思想让 LMs 解决问题时也参照类似的思路，因 LMs 在解决一般问题的方法存在两个关键缺点：
1. 局部上看，在解决问题时“一条路走到黑”，不会在思考过程探索不同的可能性；
2. 全局上看，缺乏规划(planning)、预测(lookahead)及回溯(backtracking)，进而无法辅助评估不同的决策。


为了解决这些缺点，我们引入了“思维树”（Tree of Thoughts，ToT）的范式，允许语言模型在思考过程中探索多条推理路径。ToT将任何问题框架化为对树的搜索，其中每个节点都是表示具有输入和迄今为止的思考序列的部分解决方案的状态 $$s=[x, z_{1,...,i}]$$。ToT的具体实例涉及回答四个问题：
1. 如何将中间过程分解为思维步骤；
2. 如何从每个状态生成潜在的 thoughts；
3. 如何启发式地评估状态；
4. 使用什么搜索算法。

## 思维分解
CoT 在没有显式地分解的情况下连贯地采样思维，ToT 利用问题属性设计和分解中间思维步骤。根据不同的问题，一个思维可能是几个词（填字游戏），一行方程式（24点游戏），或整个写作计划的段落（创意写作）。总体而言，一个思维应该足够“小”，以便语言模型生成有前景且多样的样本（例如，生成整本书通常太“大”而不连贯），但又足够“大”，以便语言模型评估其在问题解决方面的前景（例如，生成一个 token 通常太“小”以评估）。

## 思维生成 
给定一个树的状态 $$s=[x, z_{1,...,i}]$$，有两种策略生成下一个思维步的 $$k$$ 个候选状态：
1. 从 CoT prompts 独立同分校 i.i.d. 采样。$$z^{(j)} \sim p_\theta^\text{CoT}(z_{i+1} \vert s)$$。
2. 通过 "propose prompt" 顺序 propose thoughts 😕。 

## 状态评估 $$V(p_\theta, S)$$
通过使用 LM 来有意识地思考状态。在适用时，这种有意识的启发式比规则硬编码更灵活，并且比学习一个模型使用更少的样本。与思维生成器类似，我们考虑独立或一起评估状态的两种策略：
1. 独立地对每个状态进行价值评估，$$V(p_\theta, S)(s) \sim p_\theta^\text{value}(v \vert s), \forall s \in S$$，使用一个 value prompt 为每个状态打分，分值从 1 到 10，或者分类：e.g., sure/likely/impossible。
2. 通过不同状态的投票：$$V(p_\theta, S)(s) \sim \mathbb{I}[s = s^*]$$，其中 $$s^* = \argmax_s \sum_i \mathbb{I}[s^{(i)}=s]$$。适用场景，段落连续性。


## 搜索算法
1. 广度优先搜索（BFS）：在每一步，保留一组最有希望的状态，这用于24点游戏和创意写作，其中树的深度受限（$$T\leq 3$$），并且初始思考步骤可以被评估并剪枝为一个小集合（ $$b\leq 5$$）

2. 深度优先搜索（DFS）：首先探索最有前途的状态，直到达到最终输出 $$t \ge T$$ 或状态评估器认为从当前状态$$s$$无法解决问题。在后一种情况下，$$s$$ 的子树被修剪掉。在这两种情况下，DFS会回溯到 $$s$$ 的父状态以继续探索。

# 效果


# 代价
根据 prompt 的设计及搜索算法的不同，ToT 通常生成的 tokens 数量是 CoT 的 5 - 100 倍，Game of 24 及 Creative Writing 的实验花费了 106 刀。在 Game of 24 上平均每个 case 大概消耗 6.9k 个 tokens。因此研究人员也推荐，仅当任务需要精细推理且 CoT 效果不好时，才使用 ToT。


