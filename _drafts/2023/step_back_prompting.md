---
layout:     post
title:      Step back prompting
date:       2023-12-02
tags:   [prompting, llm]
categories: 
- nlp
---

关于 Prompting 的技术是层出不穷，有经典的 [CoT](https://arxiv.org/abs/2201.11903)，
催眠式的 "Take a deep breath and work on this problem"(TDB，[LARGE LANGUAGE MODELS AS OPTIMIZERS](https://arxiv.org/pdf/2309.03409.pdf))，也有风格清奇的 “PUA” ([Large Language Models Understand and Can be Enhanced by Emotional Stimuli](https://arxiv.org/abs/2307.11760)) 技术，最近 Google DeepMind 团队又提出来一个 [Step back prompting](https://arxiv.org/pdf/2310.06117.pdf) 技术，通过先抽象+再回答的方式进行推理。
关键的两步 ：
1. **Abstraction**，提示模型把具体问题抽象化，提取出抽象化问题的相关事实； 
2. **Reasoning**，利用抽象化问题的相关事实，提示模型进行推理，得到最终的答案。

<figure style="text-align: center">
    <img src="https://image.ddot.cc/202312/step_back_prompting_20231203_0819.png" width=645pt>
    <figcaption>Step back prompting</figcaption>
</figure>

这一思路乍一看，有点像 [RAG]({{site.baseurl}}/2023/11/16/Retrivial-augmented-generation/)，不同点 RAG 的思路是使用问题先检索，再生成，而 Step back prompting 是先抽象，再推理。这两种思路都是在原始的输入上进行改造，从而提升模型的效果。

- [ ] 模型如何训练的？
    - [ ] step back question 如何生成的？人工构造还是模型生成？
- [ ] 效果如何 ？
- [ ] 跟 RAG 的区别是什么？为什么比 RAG 好？

# 如何实现的？
LLM 对一些问题需要大量的细节，比如一道物理题：“*如果温度增加 2 倍，体积增加 8 倍，理想气体的压强 $P$ 会发生什么变化？*”。如果直接推理，LLM 可能不会推理到要使用[理想气体状态方程](https://zh.wikipedia.org/zh-cn/%E7%90%86%E6%83%B3%E6%B0%94%E4%BD%93%E7%8A%B6%E6%80%81%E6%96%B9%E7%A8%8B) $pV=nRT$。**Step-back** 要做的事情是让模型把问题抽象化？？？，即先问一个抽象的问题“这道题背后的物理原理是什么？”。
前提，step-back question 的难度低于原始的问题。