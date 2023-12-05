---
layout:     post
title:      Step back prompting
date:       2023-12-02
tags:   [prompting, llm]
categories: 
- nlp
---

关于 Prompting 的技术层出不穷，前面有经典的 [CoT](https://arxiv.org/abs/2201.11903)，
有催眠式的 "Take a deep breath and work on this problem"(TDB，[LARGE LANGUAGE MODELS AS OPTIMIZERS](https://arxiv.org/pdf/2309.03409.pdf))，也有风格清奇的 “PUA” 式 emontional stimuli ([Large Language Models Understand and Can be Enhanced by Emotional Stimuli](https://arxiv.org/abs/2307.11760))，最近 Google DeepMind 团队又提出来一个 [Step back prompting](https://arxiv.org/pdf/2310.06117.pdf) 技术，通过对问题先抽象+再回答的方式进行推理。

Step back 有关键的两步 ：
1. **Abstraction**，提示模型把具体问题抽象化，提取出抽象化问题的相关事实； 
2. **Reasoning**，利用抽象化问题的相关事实，提示模型进行推理，得到最终的答案。

<figure style="text-align: center">
    <img src="https://image.ddot.cc/202312/step_back_prompting_example_20231205_0827.png" width=789pt>
    <figcaption>Step back prompting</figcaption>
</figure>

思路乍一看，有点类似 [RAG]({{site.baseurl}}/2023/11/16/Retrivial-augmented-generation/)，不同之处在于 RAG 的方法是使用问题先检索相关文档，再结合文档与问题生成答案，而 Step back prompting 是先对问题做抽象，生成出相关背景知识及原理，再结合原问题进行推理。

# 如何实现的？
LLM 对一些问题需要大量的细节，比如一道关于气体状态的物理题：“*如果温度增加 2 倍，体积增加 8 倍，理想气体的压强 $P$ 会发生什么变化？*”。如果直接对这个问题进行推理，LLM 可能不会想到要使用[理想气体状态方程](https://zh.wikipedia.org/zh-cn/%E7%90%86%E6%83%B3%E6%B0%94%E4%BD%93%E7%8A%B6%E6%80%81%E6%96%B9%E7%A8%8B) $pV=nRT$。Step-back prompting 做的事情是先 prompt 模型去“思考”这道题考察的是什么内容，背后的原理是什么。迫使 LLM 从更深层、宽泛的角度去看待问题，提取出相关基础概念、原理及相关事实，再结合这些知识去对原问题进行推理。


## step-back question 如何生成 
通过 **few shot examples** 获得 step-back question。以 QA 为例，首先提示模型给出一些问题的 step-back question，然后在提示下面给出最多 5 条示例，即 `(original question, step-back question)` 对（如下图所示）。

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/step_back_prompt_20231204_0756.png" width=678pt>
    <figcaption> Prompt for step-back question </figcaption>
</figure>

文中对比了 QA 任务下 `PaLM-2L`, `PaLM-2L + Step-Back`, `PaLM-2L + Step-Back + RAG` 的效果，其中 `+RAG` 的效果最好。 在获得 step-back question 之后 ，会通过 RAG 召回 original question 与 step-question 相关的一些 facts，再与 original question 拼接后进行推理，得到最终的答案。Prompt 示例如下图所示。 


<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/final_qa_prompt_20231204_0816.png" width=678pt>
    <figcaption> Prompt for step-back question </figcaption>
</figure>

[MMLU](https://paperswithcode.com/dataset/mmlu) 高中物理、化学任务的处理流程比较类似，也是通过 few show samples 让 LLM 对问题进行抽象，不同点是 MMLU 任务中 abstraction 步骤获得的是问题相关的原理（first principles），最终给出  LLM 的 prompt 是 original question + first principles。


# 效果

- 在 MMLU 物理、化学任务中较 `PaLM-2L` 分别提升了 6.8 及 10.9 个点，且效果均优于 GPT-4；
- 在 TimeQA, MuSiQue, StrategyQA 上表现类似。特别地，在 TimeQA 上较 `PaLM-2L` 提升了 37.2 个点，且比 CoT 及 GPT-4 都有 30+ 点的提升。 


<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/step_back_prompting_performance_20231204_2152.png" width=678pt>
    <figcaption> Step-back prompting performance </figcaption>
</figure>

笔者在线下使用 ChatGPT(version: 2023.12.05) 做了一下实验，以上面的 MMLU 物理任务为例，让 ChatGPT 直接回答原始问题，得到的答案是气体压力会增加 2 倍，与正确答案不符合。
```text
... 
This means that if the temperature is increased by a factor of 2 and the volume is increased by a factor of 8, the pressure of the ideal gas will also increase by a factor of 2.
```

而使用 step-back prompting 的方式，ChatGPT 在 factor 数值（`4`）上推理有提升，但在最后一步把因子放错了位置，应该是 $P_1 = 4P_2$，导致最终的答案错误。

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/chatgpt_step_back_20231205_0858.png" width=678pt>
    <figcaption> Step-back prompting on ChatGPT </figcaption>
</figure>


# 对比 RAG
相似点都是在回答最终问题前，先对原始问题进行挖掘，获得更多相关信息。

不同点是 `PaLM-2L + RAG` 是通过检索 wikipedia 的方式获得相关的事实知识，再将知识与答案一并交给 LLM，检索使用的是 LLM 外部的知识。而 step back prompting 是通过 few shot learning 先回答一个更抽象的问题，从 LLM 获取与 original qustion 相关的背影知识。