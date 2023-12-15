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

人类在解决问题会采用的一种思路是，将问题拆成若干个过程，每一个过程都有多个解决方案，这些过程与方案构成一个树状结构，人在这棵树上进行搜索，逐渐找到最终的解决路径。 选择哪个分支是由启发式决定的，这些启发式有助于确定问题空间并引导问题解决者找到解决方案。

ToT 的思想让 LMs 解决问题时也参照类似的思路，因 LMs 在解决一般问题的方法存在两个关键缺点：
1. 局部上看，在解决问题时“一条路走到黑”，不会在思考过程探索不同的可能性；
2. 全局上看，缺乏规划(planning)、预测(lookahead)及回溯(backtracking)，进而无法辅助评估不同的决策。


为了解决这些缺点，我们引入了“思维树”（Tree of Thoughts，ToT）的范式，允许语言模型在思考过程中探索多条推理路径。ToT将任何问题框架化为对树的搜索，其中每个节点都是表示具有输入和迄今为止的思考序列的部分解决方案的状态 $s=[x, z_{1,...,i}]$。ToT的具体实例涉及回答四个问题：
1. 如何将中间过程分解为思维步骤；
2. 如何从每个状态生成潜在的 thoughts；
3. 如何启发式地评估状态；
4. 使用什么搜索算法。

## 思维分解 / Thoughts Decomposition
CoT 在没有显式地分解的情况下连贯地采样思维，ToT 利用问题属性设计和分解中间思维步骤。根据不同的问题，一个思维可能是几个词（填字游戏），一行方程式（24点游戏），或整个写作计划的段落（创意写作）。总体而言，一个思维应该足够“小”，以便语言模型生成有前景且多样的样本（例如，生成整本书通常太“大”而不连贯），但又足够“大”，以便语言模型评估其在问题解决方面的前景（例如，生成一个 token 通常太“小”以评估）。

## 思维生成 / Thoughts Generation
给定一个树的状态 $s=[x, z_{1,...,i}]$，有两种策略生成下一个思维步的 $k$ 个候选状态：
1. 从 CoT prompts 独立同分校 i.i.d. 采样。$z^{(j)} \sim p_\theta^\text{CoT}(z_{i+1} \vert s)$。
2. 通过 "propose prompt" 顺序 propose thoughts $[z^{(1)}, ..., z^{(k)}] \sim p_\theta^\text{propose}(z_{i+1}^{(1..k)} \vert s)$。😕

## 状态评估 $V(p_\theta, S)$ <span id="state-evaluate"></span>
通过 LLM + few shots 启发式的选择后续路径的开始状态，这种方法比规则硬编码更灵活，也不需要专门训练一个模型。与思维生成类似，研究人员也设计了两种策略：
1. 对每个状态分别进行评估，$V(p_\theta, S)(s) \sim p_\theta^\text{value}(v \vert s), \forall s \in S$，使用一个 value prompt 为每个状态打分，分值从 1 到 10，或者返回分类结果：e.g., `sure/likely/impossible`，然后再转化成数值。
2. 通过不同状态的投票：$V(p_\theta, S)(s) \sim \mathbb{I}[s = s^*]$，其中 $s^* = \argmax_s \sum_i \mathbb{I}[s^{(i)}=s]$。适用场景，段落连续性。

这里有个问题，state 跟 thought 有什么关系 ？

Thought 是指导 LLM 一步步去解决问题的中间步骤/结果，可以是一条公式，或者一条写作计划，在论文里面不同的 thought 用 $z_i$ 表示。而 state 指的是由 input 及一系列 thoughts 构成的解题路径，在论文里用 $[x, z_{1,...,i}]$ 表示。state 的评估是为了判断当前 thought 是否可行，是否有前景，是否应该继续探索下去。

**思维生成与状态（可行性）评估是 ToT 最重要的两块**，任何一个效果不佳，对模型解题都会带来困难。对于需要精细推理（deliberate reasoning）的问题类型来说更是如此。通俗一点说，思维生成和搜索算法用于探索可行的解题路径，而状态评估用于鉴别哪些路径是可行的。

笔者在线下测试 Game of 24 时在这两点都遇到了问题。笔者使用 ChatGPT 及 Gemini（Google 最近刚公开 [Gemini API](https://ai.google.dev/)）分别做了一下测试。总的来说，ChatGPT 在**思维生成上表现不错，但在状态评估上表现不佳**。实验中发现，给定合适的 [prompt](https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/prompts/game24.py#L52)，ChatGPT 几乎能完美生成下一步所有可能状态的集合，但对状态打分结果比较差，可行的状态给了低分，不可行的状态却给了高分，导致 $b=1$ 时大概率陷入到一条不可行的路径上。比如下图中的 Game of 24 示例中，最有可能的状态 `4 5 4 (4 * 5 + 4 = 24)` 给的打分是 1，而不可行的状态 `6 9 10` 给的打分是 20。而 Gemini 在**思维生成上表现较差**，在 `propose_prompt` 1-shot 效果比 ChatGPT 差了很多，生成的思维里会多次复用数字，在状态评估上也没有明显优势。 


<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/bad_state_evaluator_20231214_1134.png" width=789pt>
    <figcaption style="text-align:center">Bad state evaluator</figcaption>
</figure>


## 搜索算法
1. 广度优先搜索（BFS）：在每一步，保留一组最有希望的状态，这用于24点游戏和创意写作，其中树的深度受限（$T\leq 3$），并且初始思考步骤可以被评估并剪枝为一个小集合（ $b\leq 5$）

2. 深度优先搜索（DFS）：首先探索最有前途的状态，直到达到最终输出 $t \ge T$ 或状态评估器认为从当前状态$s$无法解决问题。在后一种情况下，$s$ 的子树被修剪掉。在这两种情况下，DFS会回溯到 $s$ 的父状态以继续探索。

# 效果

## Game of 24
1362 个问题集合见 [4nums.txt](https://image.ddot.cc/202312/4nums.txt)，从上到下难度（解题中位时间，median time）依次递增。研究人员使用的 [IO prompt](https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/prompts/game24.py) 5-shots 示例:

```python
# 5-shot
standard_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
'''
```

对应的 CoT prompt 在 IO prompt 的基础上，增加了 3 条中间步骤的提示，如下表所示。这个 CoT prompt 看起来像是提示了一个解题的思路，但实际上增加了解题的难度，因为模型需要在这个思路的基础上进行推理，每一步还都要正确。过程不正确但结果正确是有可能的，但概率极低，更可能的情况是过程错误，结果也错误。

```python
# 5-shot
cot_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24
... # 4 more examples
Input: {input}
'''
```

ToT 的实验流程如下图所示：

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/tot_game_24.png" width=989pt>
    <figcaption style="text-align:center">Game of 24 实验流程</figcaption>
</figure>

### 结果 

使用 GPT-4，ToT + BFS($b=1$) 可以达到 45% 的成功率，当 $b=5$ 时，可以达到 74% 的成功率，作为对比
- IO prompt, CoT prompt, CoT-SC 的成功率分别为 7.3%, 4%, 9%。
- IO best 100(100 次成功一次) 成功率提升到 33%，CoT best 100 成功率提升到 49%。

### 实验复现
笔者在线下用 ChatGPT(2023.12.13)做了一下测试（GPT-4 太贵了），IO prompt 三次成功率分别为 0.07, 0.11, 0.08，平均成功率为 8.6%，CoT prompt 三次成功率分别为 0.05, 0.04, 0.07，平均成功率为 5.3%，跟论文中的结果差不多。

但 ToT $b=1$ 的三次结果分别为 0.06, 0.11, 0.07，平均成功率为 0.08，跟论文中的 0.45 相差较大，甚至不如 IO prompt。

当然，一个原因可能是笔者使用的 ChatGPT 而不是 GPT-4，效果上有折扣。如上面[状态评估](#state-evaluate)小节里分析，ChatGPT 在<span style="color:red">状态评估阶段效果并不理想</span>。比如给定第一次思维分解后的状态 `3 4 5 7\n7-5=2(left: 3 4 2)`，通过 state evaluate 判断凑成 24 的可能性是 `sure`、`likely` 还是 `impossible`，ChatGPT 在 30 个结果都为 `sure` 的[状态](https://dlj.one/h9m6gl)上 3 次平均正确率为 0.17，也即是错误率为 0.83。

因为 state evaluate 这一步也是通过 LLM 来实现的，本身就很依赖 LLM 本身的理解推理能力。像 Game of 24 的实验设计里，第一步 thought decomposition 之后，数字从 4 个数减少为 3 个数，然后让 LLM 判断这 3 个数能否凑够 24， 这个问题在难度上比原问题其实并没有降低多少。

研究人员还统计了 CoT，ToT 失败步数比例，失败指走到某一步无法继续推理下去，比如所有 thoughts 都是 invalid。如下图 Figure 3(a) 所示，CoT 的失败情况中有超过 60% 的比率在第一步就错了，ToT($b=5$) 的失败情况都集中在最后一步。对于基于树的搜索算法来说，这是结果并没有让人意外的地方，当 $b=5$ 时 ToT 理论上可能考虑 $5^4=625$ 种可能的路径，而 CoT 只生成一条路径，因此半道折戟的概率会更高。

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/tot_game_24_20231213_1125.png" width=789pt>
    <figcaption style="text-align:center">Game of 24 结果对比</figcaption>
</figure>


# 代价
根据 prompt 的设计及搜索算法的不同，ToT 通常生成的 tokens 数量是 CoT 的 5 - 100 倍，Game of 24 及 Creative Writing 的实验花费了 106 刀。在 Game of 24 上平均每个 case 大概消耗 6.9k 个 tokens。因此研究人员也推荐，仅当任务需要精细推理且 CoT 效果不好时，才使用 ToT。


