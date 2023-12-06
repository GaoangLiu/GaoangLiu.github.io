---
layout:     post
title:      Step-Back Prompting
date:       2023-12-02
tags:   [prompting, llm]
categories: 
- nlp
---

Prompting 技术层出不穷，前面有经典的 [CoT](https://arxiv.org/abs/2201.11903)，有催眠式的 "Take a deep breath and work on this problem"(TDB，[LARGE LANGUAGE MODELS AS OPTIMIZERS](https://arxiv.org/pdf/2309.03409.pdf))，也有风格清奇的 “PUA” 式 emontional stimuli ([Large Language Models Understand and Can be Enhanced by Emotional Stimuli](https://arxiv.org/abs/2307.11760))，其他的还有:
- ToT: [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
- SC: [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
- Multi-Persona: [Unleashing Cognitive Synergy in Large Language Models: A Task-Solving Agent through Multi-Persona Self-Collaboration](https://arxiv.org/abs/2307.05300)
- Least to Most: [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625)
- ART: [ART: Automatic multi-step reasoning and tool-use for large language models](https://arxiv.org/abs/2303.09014)
- ReART: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- Reflextion: [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- HyDE: [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)

最近 Google DeepMind 团队又提出了 [Step-Back Prompting](https://arxiv.org/pdf/2310.06117.pdf) 技术，通过对问题先抽象+再回答的方式进行推理。Step back 有关键的两步 ：
1. **Abstraction**，提示模型把具体问题抽象化，先提取出更基础、更高层的事实知识； 
2. **Reasoning**，利用相关事实，提示模型进行推理，得到最终的答案。

<figure style="text-align: center">
    <img src="https://image.ddot.cc/202312/step_back_prompting_example_20231205_0827.png" width=789pt>
    <figcaption>step-back prompting</figcaption>
</figure>

思路乍一看，有点类似 [RAG]({{site.baseurl}}/2023/11/16/Retrivial-augmented-generation/)，不同之处在于 RAG 的方法是使用问题先检索相关文档，再结合文档与问题生成答案，而 Step Back Prompting 是先对问题做抽象，使用 LLM 生成出相关背景知识及原理，再结合原问题进行推理。

# 如何实现
LLM 解决一些问题时需要大量的细节，考虑一道关于气体状态的物理题：

>如果温度增加 2 倍，体积增加 8 倍，理想气体的压强 $P$ 会发生什么变化？

如果直接对这个问题进行推理，LLM 可能不会想到要使用[理想气体状态方程](https://zh.wikipedia.org/zh-cn/%E7%90%86%E6%83%B3%E6%B0%94%E4%BD%93%E7%8A%B6%E6%80%81%E6%96%B9%E7%A8%8B) $pV=nRT$。Step-back prompting 做的事情是先 prompt 模型去“思考”这道题考察的是什么内容，背后的原理是什么。迫使 LLM 从更深层、更宽泛的角度去看待问题，提取出相关基础概念、原理及知识，再结合这些知识去对原问题进行推理。


## Step-back question 如何生成 
通过 **few shot examples** 获得 step-back question。

以 QA 为例，首先提示模型给出一些问题的 step-back question，然后在提示下面给出最多 5 条示例，即 `(original question, step-back question)` 对（如下图所示）。

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/step_back_prompt_20231204_0756.png" width=678pt>
    <figcaption> Prompt for step-back question </figcaption>
</figure>

示例的个数不用很多，文中做了 ablation 实验，发现一个示例就足以让 LLM 学习到抽象的能力。生成 step-back question 只是第一步，重要的是生成合适的 step-back question，并通过这个 question 得到必要的信息。对于 QA 这种知识密集型的任务，仅靠 LLM 本身的“知识储备”可能还不够，需要借助外部知识库，比如 wikipedia，来获得更多的信息。

文中对比了 QA 任务下 `PaLM-2L`, `PaLM-2L + Step-Back`, `PaLM-2L + Step-Back + RAG` 的效果，其中 `+RAG` 的效果最好。 在获得 step-back question 之后 ，会通过 RAG 召回 original question 与 step-question 相关的一些 facts，再与 original question 拼接后进行推理，得到最终的答案。Prompt 示例如下图所示。 


<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/final_qa_prompt_20231204_0816.png" width=678pt>
    <figcaption> Prompt for step-back question </figcaption>
</figure>

[MMLU](https://paperswithcode.com/dataset/mmlu) 高中物理、化学任务的处理流程比较类似，也是通过 few show samples 让 LLM 对问题进行抽象，不同点是 MMLU 任务中 abstraction 步骤获得的是问题相关的原理（first principles），最终给出  LLM 的 prompt 是 original question + first principles。


# 效果
在实验环节，研究人员对以下不同种类的任务进行了实验：
1. STEM。数据集：MMLU 高中物理、化学部分，这部分任务需要深度推理。
2. 知识 QA。数据集：TimeQA，包含时间敏感的复杂问题，SituatedQA，要求根据时间或地理背景回答问题。
3. 多跳推理。数据集：MuSiQue 及 StrategyQA。

实验的基准模型是 `PaLM-2L`，其它参与对比的模型及方案有 `PaLM-2L + 1-shot`、`PaLM-2L + CoT`、`PaLM-2L + CoT + 1-shot`、`PaLM-2L + TDB`、`PaLM-2L + RAG` 等。结果评估阶段，使用 few-shot prompt `PaLM-2L` 来判断推理答案与真实答案是否一致。

## STEM
- 在 MMLU 物理、化学任务中 `PaLM-2L + Step-Back` 较 `PaLM-2L` 分别提升了 6.8 及 10.9 个点，且效果均优于 GPT-4；

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/step_back_prompting_performance_20231204_2152.png" width=678pt>
    <figcaption> Step-back prompting performance </figcaption>
</figure>

笔者在线下使用 ChatGPT(version: 2023.12.05) 上面的*理想气体状态*任务做了一下实验， 发现 step-back 确实有帮助。
首先让 ChatGPT 直接回答原问题，得到的答案是“气体压力会增加 2 倍”，这与正确答案不符合。
```text
... 
This means that if the temperature is increased by a factor of 2 and the volume is increased by a factor of 8, the pressure of the ideal gas will also increase by a factor of 2.
```

然后使用 step-back prompting 的方式，ChatGPT 在 factor 数值（`4`）上推理有提升，但在最后一步把因子放错了位置（对应于文中的 *Math Error*，正确结果应该是 $P_1 = 4P_2$），进而在最后推理时也出现错误。

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/chatgpt_step_back_20231205_0858.png" width=678pt>
    <figcaption> Step-back prompting on ChatGPT </figcaption>
</figure>

## Knowledge QA

- 在 TimeQA 上较 `PaLM-2L` 提升了 37.2 个点，且比 CoT 及 GPT-4 都有 30+ 点的提升。 

在这个实验阶段，考虑到任务的知识密集性，研究人员同时使用了 RAG 与 step-back prompting，即 original question, step-back question 通过 RAG 检索到的段落与 original question 拼接在一起，再让 LLM 进行推理。
1. `PaLM-2L`, `GPT-4` 分别得到 41.5%, 45.6% 的准确率，这说明任务本身还是非常难的。 
2. 在基线模型上应用 CoT 或 TDB，准确率相对上面两个方案还下降了一点。
3. `PaLM-2L + RAG` 将准确率提升到 57.4%，凸显了任务的知识密集性。
4. `PaLM-2L + RAG + Step-Back` 将准确率提升到 68.7%，说明进行抽象再检索这一步是非常用效的。


## Multi-Hop Reasoning

- 在 MusiQue 上 `PaLM-2L + Step-Back` 方案以 42.8% v.s., 35.5% 超出 `PaLM-2L` 7.3 个百分点，较 `PaLM-2L + CoT` 的 38.7% 也有 4.1 个点的提升。 在 StrategyQA 上的表现类似。 


# 消融实验
针对任务 STEM tasks, Knowledge QA, Multi-Hop Reasoning 都分别做了 ablation analysis，总的结论是 step-back prompting 这个策略本身对效果有较大的提升。

## [STEM tasks](https://arxiv.org/pdf/2009.03300.pdf)

- Few shot ablation。在 MMLU 物理任务中，few shot 示例个数从 1 到 5 增长时，模型表现没有明显差异。说明检索相关原理和概念的任务相对容易学习，一个示例就足够了。 
- 错误分析。仍以 MMLU 物理任务为基准，abstraction 阶段的错误，即提取的 principle 错误或者不完整（*Principle Error*）的情况，仅占所有错误类型中的 9%，而超过 90% 的错误都发生在 reasoning 阶段，这个阶段里的错误包括*事实错误*、*数学错误*、*上下文丢失*及*推理错误*。这表明在复杂任务推理过程中，错误产生的主要原因在于**模型本身推理能力的局限性**，而非 step-back prompting 技术上的缺陷。这里面 Math Error 及 Reasoning Error 的总占比高达 80%，这说明对这一类任务 LLM 的数学及推理能力仍然是解决问题的关键。 

## Knowledge QA
- Few shot ablation。在 TimeQA 上实验结果上面 MMLU 一致，一个示例就足以让 `PaLM-2L` 学习到 abstraction skills。
- 错误分析。
    - 在所有错误中，step-back 错误，即产生了 not-helpful 的 step-back question 的情况占比很低，仅为 1%，而有一半以上的错误源于推理错误，45% 的错误源于 RAG 没有召回到相关信息。
    - 作者对比了 `PaLM-2L + RAG` 与 `PaLM-2L + RAG + Step-Back`，后者修正了前者 21.6% 的错误，彰显了在做 Knowledge QA 问题前进行抽象的必要性。 

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/ablation_step_back_prompting_20231205_1040.png" width=649pt>
    <img src="https://image.ddot.cc/202312/error_analysis_of_step_back_qa_20231205_1202.png" width=589pt>
    <figcaption> Ablation and error analysis of step-back prompting on TimeQA. </figcaption>
</figure>


# 对比 RAG
相似点都是在回答最终问题前，先对原始问题进行挖掘，获得更多相关知识，提升最终推理的准确性。

不同点是 `PaLM-2L + RAG` 是通过检索 wikipedia 的方式获得相关的事实知识，再将知识与答案一并交给 LLM，检索使用的是 LLM 外部的知识。而 step-back prompting 是通过 few shot learning 先回答一个更抽象的问题，从 LLM 获取与 original qustion 相关的背影知识。

从研究人员在 TimeQA 上的实验结果中可以看到，在简单任务上，RAG, step-back 都能得到不少的提升。对于困难任务，RAG 提升不明显，而 step-back（不带 RAG）的提升仍然比较明显。作者没有给出具体的结果差异，但笔者猜测，step-back 通过 LLM 检索得到更高层、全面的知识，而 RAG 检索获得的还是偏细节的内容，从而缺少推理准确答案需要的额外信息。 

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/timeqa_easy_hard_20231205_1446.png" width=749pt>
    <figcaption> Strong performance of step-back prompting on Knowledge QA tasks. </figcaption>
</figure>
