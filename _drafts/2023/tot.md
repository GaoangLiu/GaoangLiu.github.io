---
layout:     post
title:      Tree of Thoughts
date:       2023-12-10
tags:   [prompt, tot]
categories: 
- nlp
---
[Tree of thoughts(ToT)](https://arxiv.org/abs/2305.10601) 是由普林斯顿大学和谷歌 DeepMind 联合提出的一种新型模型推理框架，是 [Chain of Thoughts Prompting](https://arxiv.org/abs/2201.11903)(CoT) 的泛化形式。ToT 将语言模型的推理过程建模为对树状结构的搜索，通过逐步将问题拆解为更易处理的子问题，每一步都探索并缓存可行的解题路径，从而提高语言模型解决问题的能力。更多详细内容可参考论文： [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)。


<figure style="text-align: center">
    <img src="https://image.ddot.cc/202312/various_approaches_20231212_1609.png" width=789pt>
    <figcaption>图1：ToT 与其他语言模型推理框架的比较</figcaption>
</figure>


# 概览

CoT 的提出旨在解决输入 $x$ 到输出 $y$ 的映射关系非平凡（non-trivial）的情况，即那些难以通过简单的一步推理就能解决的问题，典型代表为数学和逻辑问题。CoT 的核心思想是从 $x$ 到生成 $y$ 的过程中，引入一系列 thoughts $z_i$ 以协助模型进行推理。对于数学问题而言，$z_i$ 可以表示解题时的中间步骤。 形式化地，CoT: $y \sim p_\theta^\text{CoT}(x, z_{1,...,n})$，其中:
- $z_i \sim p_\theta^\text{CoT}(x, z_{1,...,{i-1}})$，每一个 thought $z_i$ 都依赖于输入 $x$ 及前面的 thoughts $z_1, ..., z_{i-1}$。
- $p_\theta$ 表示参数为 $\theta$ 的预训练语言模型，$p_\theta(x) = \prod_{i=1}^n p_\theta(x_i \vert x_{<i})$。 

CoT-SC（Self-consistency with CoT）是基于 CoT 的一种集成策略，通过考虑并综合多个独立同分布（i.i.d）的 CoT 结果来进行最终的决策。其形式化表示为：$[z_{1, ...,n}^{(i)}, y^{(i)}] \sim p_\theta^\text{CoT}(z_{1,...,n}, y \vert x)$，最终输出为出现频率最高的结果 $y=\argmax_y{\sum_i \mathbb{I}(y^{(i)}=y)}$。CoT-SC 相较于朴素的 CoT 更为有效的原因在于，同一个问题通常存在多种不同的解题路径，**通过探索更多的解题路径，并对最终结果进行投票表决，有望提高输出决策的可靠性**。这种方法特别适用于输出结果有限的情况，例如多选问答。

CoT-SC 存在的一个问题是，它探索的这些路径除了起点一致、在终点参与决策以外，彼此之前是独立的、没有关联的。 人类在解决问题时常采用将问题拆解成若干个步骤的思路，每一步都考察多个解决方案，这些过程与方案构成一个树状结构。解决问题相当于在这棵树上进行搜索，逐渐找到一条可行的解决路径。在搜索过程中，人们会借助一些启发式策略来决定选择哪个分支。ToT 的思想旨在让语言模型（LMs）解决问题时也采用类似的思路，从而弥补 LMs 存在的两个关键不足：

1. 局部性：LMs 在解决问题时通常“一条路走到黑”，不会思考探索不同的可能性；
2. 全局性：LMs 缺乏整体规划、预测（lookahead）及回溯（backtracking）的能力，从而无法辅助评估不同的决策。

ToT 将问题框架化为对树的搜索，其中节点表示由输入 $x$ 和思考 $z_i$ 构成的状态 $s=[x, z_{1,...,i}]$。一般而言，ToT 的工作流程包括以下几个步骤：

- **问题拆解**： 将问题分解为若干个步骤。
- **生成中间结果**： 在每个步骤中生成所有可能的中间结果，即 thoughts。
- **启发式评估**： 以一种启发式方法评估这些中间结果（在文中被称为状态评估）。
- **搜索算法探索**： 结合广度优先搜索（BFS）、深度优先搜索（DFS）等搜索算法，持续探索状态直到达到最终步骤。

ToT 与 CoT-SC 的主要差异在于前者以一种层次化的方式解决问题，后者强调的是结果的一致性。

## 思维分解 / Thoughts Decomposition
在这里，“Thought” 指的是语言模型（LM）在解题过程中生成的中间结果，该结果用于辅助 LM 进行后续的思考与推理。

> ToT actively maintains a tree of thoughts, where each thought is a coherent language sequence that serves as an intermediate step toward problem solving...

以 Game of 24 为例，给定输入 `2 2 10 1`，经过一步计算 `1*2=2` 之后，这个中间结果可以是 `left:2 2 10`。对于创意写作，中间结果可以是由如下 prompt 得到一个段落的写作计划。

```python
cot_prompt = '''
Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {input}

Make a plan then write. Your output should be of the following format:

Plan:
Your plan here.

Passage:
Your passage here.
'''
```

Thought 的设计需要在粒度上做出权衡，保证语言模型既能产生富有创意和多样性的输出，又能够提供足够的信息，支持问题的深入解决。因此，thought 需要“足够小”，使得语言模型能够灵活地生成各种可能性，又要“足够大”，使得语言模型能够全面地评估其在整个问题解决过程中的潜在价值。

## 思维生成 / Thoughts Generation
针对给定状态 $s=[x, z_{1,...,i}]$，ToT 提出了两种用于生成下一步 $k$ 个 thoughts 的策略：
1. **独立同分布采样策略**： 从 CoT prompting 结果中进行独立同分布采样，$z^{(j)} \sim p_\theta^\text{CoT}(z_{i+1} \vert s)$。
2. **propose 生成策略**：通过 "propose prompt" 生成 thoughts $[z^{(1)}, ..., z^{(k)}] \sim p_\theta^\text{propose}(z_{i+1}^{(1..k)} \vert s)$。😕

## 状态评估 $V(p_\theta, S)$ <span id="state-evaluate"></span>
通过 Language Model（LLM）+ Few Shots 启发式的选择可行状态。启发式方法相较于规则硬编码更为灵活，而且无需专门训练一个模型。与思维生成类似，研究人员设计了两种评估策略：
1. **逐一评估**：对每个状态分别进行评估，$V(p_\theta, S)(s) \sim p_\theta^\text{value}(v \vert s), \forall s \in S$，使用一个 LM 为每个状态打分，分值从1到10，或者返回分类结果，例如 sure/likely/impossible，然后再转化成数值。
2. **状态投票策略**：通过不同状态的投票进行评估 $V(p_\theta, S)(s) \sim \mathbb{I}[s = s^*]$，其中 $s^* = \argmax_s{\sum_i \mathbb{I}[s^{(i)}=s]}$。适用场景，判断段落连续性。

在这里，有两个概念，分别是 state 与 thought。如上所述，thought 是指导 LLM 一步步解决问题的中间步骤/结果，可以是一条公式或者一条写作计划，在论文中用 $z_i$ 表示。而 state 指的是由 input 及一系列 thoughts 构成的解题路径，在论文中用 $[x, z_{1,...,i}]$ 表示。state 的评估旨在判断当前 thought 是否可行、是否有前景，以及是否应该继续探索下去。

### 思维生成与状态评估决定最终的效果
在 ToT 中，**思维生成与状态评估**是两个至关重要的步骤。任何一步效果不佳都会对模型解题带来困难，尤其是在需要进行精细推理（deliberate reasoning）的问题中。通俗地说，思维生成和搜索算法用于探索可行的解题路径，而状态评估则用于鉴别哪些路径是可行的。

笔者在线下测试 Game of 24 时在这两步都遇到了挫折。测试分别使用了 ChatGPT 及 Gemini（Google 最近公开了 [Gemini API](https://ai.google.dev/)），结论是：这两个模型结合 ToT 的效果都不理想。总的来说，ChatGPT 在**思维生成上表现优异，但在状态评估上表现不佳**。实验中发现，给定合适的 [prompt](https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/prompts/game24.py#L52)，ChatGPT 几乎能完美生成下一步所有可能（未必可行）状态的集合，但对状态打分结果比较差，可行的状态给了低分，不可行的状态却给了高分，导致 $b=1$ 时大概率陷入到一条不可行的路径上。比如下图中的 Game of 24 的一条例子，输入是 `4 5 6 10`，经过一步 thought generation，在给状态打分时， ChatGPT 给最可行状态 `4 5 4 (4 * 5 + 4 = 24)` 的打分是 1，而给不可行的状态 `6 9 10` 的打分是 20。而 Gemini 在**思维生成上表现较差**，在 `propose_prompt` 1-shot 效果比 ChatGPT 差了很多，生成的思维里会多次复用数字，在状态评估上也没有明显优势。 

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/bad_state_evaluator_20231214_1134.png" width=789pt>
    <figcaption style="text-align:center">Bad state evaluator</figcaption>
</figure>


## 搜索算法
1. 广度优先搜索（BFS）：在每一步保留一组最有希望的状态，如果状态数量过多，则剪枝（采样）成一个子集，即限制每一步树的宽度为 $b\leq B$（e..g, $B=5$）。这个算法应用到 Game of 24 及 creative writing 任务上。 
2. 深度优先搜索（DFS）：每步探索最可行的状态，直到达到最终输出 $t \ge T$，即发现一条可行路径，或者状态评估器认为当前状态 $s$ 不可行。在后一种情况下，$s$ 的子树会被修剪掉。



# 效果

## Game of 24
使用 GPT-4，ToT + BFS($b=1$) 可以达到 45% 的成功率，当 $b=5$ 时，可以达到 74% 的成功率，作为对比
- IO prompt, CoT prompt, CoT-SC 的成功率分别为 7.3%, 4%, 9%。CoT 的效果反而不如直接问 LM。
- IO best 100(100 次成功一次) 成功率提升到 33%，CoT best 100 成功率提升到 49%。

### 实验设计
1362 个问题集合见 [4nums.txt](https://image.ddot.cc/202312/4nums.txt)，从上到下难度依次递增，难度以用户解题中位时间为基准。研究人员使用的 [IO prompt](https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/prompts/game24.py) 5-shots 示例:

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

对应的 CoT prompt 是在 IO prompt 的基础上，增加了 3 条中间步骤的提示。在笔者看来，这个 CoT prompt 的设计实际上增加了解题的难度，因为模型需要在这个思路的基础上进行推理，每一步还都需要推理正确。过程不正确但结果正确是有可能的，但概率极低，更可能的情况是过程错误，结果也错误。

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


### 实验复现
笔者在线下用 ChatGPT(2023.12.13)做了一下测试（GPT-4 太贵了），IO prompt 三次成功率分别为 0.07, 0.11, 0.08，平均成功率为 8.6%，CoT prompt 三次成功率分别为 0.05, 0.04, 0.07，平均成功率为 5.3%，跟论文中的结果差不多。

但 ToT + $b=1$ 的三次结果分别为 0.06, 0.11, 0.07，平均成功率为 0.08，跟论文中的 0.45 出入较多，这个结果甚至不如 IO prompt。$b=5$ 时三次实验结果分别为 0.13, 0.23, 0.14，平均成功率为 0.167，这个结果对比 IO prompting 及 CoT 无疑有一定提升，但跟论文中的 0.74 相差还是很大。

差距较大的一个原因可能是笔者在实验中使用了 ChatGPT 而非 GPT-4，效果上有折扣。如上面[状态评估](#state-evaluate)小节里分析，ChatGPT 在**状态评估阶段效果并不理想**。为验证这一点，笔者还专门做了一个小实验，这个实验的任务跟 state evaluation 一致，即给定第一次思维生成后的状态（e.g., `3 4 5 7\n7-5=2(left: 3 4 2)`），让 ChatGPT 判断凑成 24 的可能性是 `sure`、`likely` 还是 `impossible`，实验数据是 30 条结果都应该为 `sure` 的状态，数据集见 [thoughts.txt](https://dlj.one/h9m6gl)。实验重复 3 次并求平均结果，ChatGPT 在 30 条状态评估的正确率仅为 0.17，也即是错误率在 0.8 以上。

另一个原因是，**thought 生成及状态评估并没有显著降低推理的难度**。没错，笔者对论文结果，特别是 Game of 24 的结果表示置疑。因为 state evaluation 这一步也是通过 LLM 来实现的，效果直接依赖 LLM 本身的理解推理能力。像 Game of 24 的实验设计里，第一步 thought generation 之后，输入从 4 个数减少为 3 个数，然后让 LLM 判断这 3 个数能否凑够 24，这个问题在难度上比原问题其实并没有降低多少。再延伸一下，我们考虑新的 “Game of 100”：输入不是 4 个数，而是 10 个数，目标值是 100，那么第一步 thought generation 之后，让 LLM 去评估余下的 9 个数能否凑够 100，GPT 在这一步的表现等同于盲猜这一点应该就容易理解了。结果是，如果 GPT 在状态评估阶段的表现比较差，那 ToT 的搜索算法就会陷入到一条不可行的路径上，这也是笔者在实验中遇到的情况。


研究人员还对 CoT 和 ToT（$b=5$）的失败步数比例进行了统计，其中失败表示在某一步无法继续推理下去，例如所有 thoughts 都是无效的。如图 3(b) 所示，CoT 在第一步就出现错误的比率超过了 60%，而 ToT（$b=5$）的失败情况主要集中在最后一步，前几步很少出现错误。对于基于树的搜索算法而言，这个结果并没有让人感到意外。当 $b=5$ 时，ToT 理论上可能考虑 $5^4=625$ 种可能的路径，而 CoT 只生成一条路径，因此在推理的过程中半道折戟的概率会更高。

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/tot_game_24_20231213_1125.png" width=789pt>
    <figcaption style="text-align:center">Game of 24 结果对比</figcaption>
</figure>


# 代价
ToT 对比 CoT 在效果上确实有提升，但对应的计算量的增加也是显而易见的，根据 prompt 的设计及搜索算法的不同，ToT 生成的 tokens 数量通常是是 CoT 的 5 到 100 倍。Game of 24 及 Creative Writing 的实验花费了 106 刀。在 Game of 24 上平均每个 case 大概消耗 6.9k 个 tokens。因此研究人员也推荐，仅当任务需要精细推理且 CoT 效果不好时，才使用 ToT。


