---
layout:     post
title:      Prefix Tunning
date:       2023-12-06
tags:   [gemini, llm, deepmind]
categories: 
- nlp
---

从 ChatGPT 发布之后，Google 搜索一度下降，在 AI 这一块，Google 一直都很被动，
连创始人都下场搞 AI，先后出了 BARD，但反映平平，对比 ChatGPT 没有什么惊艳之处，甚至还有很多不足，比如早期的 Bard 仅支持英文。

今天 Google DeepMind 终于搞了一个大的，发布了 [Gemini 1.0](https://deepmind.google/technologies/gemini/#bard)，多模态的 AI 语言模型，同时支持文字、图片、音频、视频等多种模态的处理，这个模型的效果非常好，可以说是目前最好的 综合 AI 语言模型了。

Gemini 这次主要对标 GPT-4，在数据、推理、代码及综合能力等方面都超过了 GPT-4，比如在 [MMLU](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu) 任务上， Gemini Ultra(CoT@8) 得分 90，而之前的 SoTA 由 GPT-4 (5-shots) 保持，得分为 86.4，这也是首次有模型在这个任务上得分超过 90 分。

<img src="https://image.ddot.cc/202312/gemini_final_text_table_bigger_font_amendment_lines.gif" width=789pt>

<img src="https://image.ddot.cc/202312/gemini_20231207_0913.png" width=789pt>


Gemini 1.0，有三个版本大小的模型：
- Gemini Ultra — 宇宙当前最强模型，适用于高度复杂的任务。
- Gemini Pro — 性能、大小和速度的最佳平衡，适用于大多数任务。
- Gemini Nano — 适合终端设备上的模型，比如在 Pixel 8 上运行。


根据 [Gemini 的报告](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf)，Gemini Pro 基本可以媲美 GPT-3.5，在 8 项行业标准基准测试中，有六项中表现优于GPT-3.5，其中包括 MMLU 及 GSM8K。 Gemini Pro 已经集成到 Bard 上（参考：[Bard gets its biggest upgrade yet with Gemini](https://blog.google/products/bard/google-bard-try-gemini-ai/)）。Gemini Ultra 会在 2024 年初集成到 Bard Advanced 上，应该就是 Google 版 ChatGPT premium 吧。 


不足：
Google 说对 Bard 中的 Gemini Pro 做了微调，使其在理解、总结、推理、编码和规划等方面更加强大。但实际使用中，理解（泛化）能力还是不太够。比如，这是 Bard 的示例:
```
Give me insights about this video: https://www.youtube.com/watch?v=lr87yrvK86w 
Organize the information in a set of easy to scan bullet points.
```

Bard 会对视频进行分析，然后生成一段文字，这段文字会包含视频的一些要点。但如果我把视频链接换成其他 YouTube 链接，比如 [Dr. Andrej Karpathy 的 State of GPT YouTube 视频](https://youtu.be/bZQun8Y4L2A?si=8D0NjPDZaLkbXW-f)，prompt 后面加上 reply in Chinese：
```
Give me insights about this video: https://youtube.com/watch?v=bZQun8Y4L2A
Organize the information in a set of easy to scan bullet points and reply in Chinese.
```

Bard 就不知所措了，只回答：
```
I'm not able to help with that, as I'm only a language model.
``` 

事实上 Bard 是支持回复中文的，上面的 prompt 多次尝试后，有机率可以得到正常回复：
```
视频“State of GPT | BRK216HFS”的见解 ...
```

Gemini Nano 其实还有两个版本，一个 Nano 1, 参数 1.8B，另一个 Nano 2，参数 3.25B。


Google 同时还放出了基于 Gemini 的 [AlphaCode 2](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf)，较上一代 AlphaCode 可以多解决 170% 的问题。


# 模型效果
支持文字、图片、音频、视频包括跨模型，且每个模态都很强，Gemini 在 12 文本和推理基准测试中的 10 项，9 项图像理解基准测试，6 项视频理解基准测试，以及 5 项语音识别和语音翻译基准测试中都刷新了 SoTA。

<img src="https://pbs.twimg.com/media/GArHxGRagAElovD?format=jpg&name=large" width=778pt>

在 [Technical Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf) 中的第一个例子里，Google 展示了 Gemini 从图片中识别学生潦草的字体，验证学生的推理过程并根据要求给出符合格式的解答。

<img src="https://image.ddot.cc/202312/gemini_example_1_20231207_1505.png" width=789pt>


## 文本
Gemini Ultra 在 MMLU 上首次超过人类水平（89.8），得分 90，比之前的 GPT-4 还要高出 3 ~ 4 个点。 这主要利益于 Gemini Ultra 使用的一个 *uncertainty-routed chain-of-thought* 策略，这个策略笔者暂时还没有完全理解，看起有点预测多个结果然后投票的意思，但细节不是很清楚（Twitter 网友 [@yi_ding](https://twitter.com/yi_ding/status/1732443815804653744) 也有同样的疑惑）。原文如下：

```text
...
We proposed a new approach where model produces k chain-of-thought samples, selects the majority vote if the model is confident above a threshold, and otherwise defers to the greedy sample choice. The thresholds are optimized for each model based on their validation split performance. 
...
```

Gemini Ultra + 5-shot 效果事实上比 GPT-4 + 5-shot 低 2.7 个点，但 Gemini Ultra + 32-shot + CoT 再加上上面的 consensus voting 的方法才终于超过了 GPT-4，Google 这次为了在 MMLU 上超越 GPT-4 可谓是“下足了功夫”。因此[有人]对 Gemini 的训练表示了质疑。 

<img src="https://image.ddot.cc/202312/satire_gemini_20231207_2130.png" width=389pt>

在 GSM8K 上达到了 94.4% 的准确率，策略是 CoT + [self-consistency](https://arxiv.org/pdf/2203.11171.pdf)，就是预测多个结果，然后投票。 


## 编程
同时还推出基于 Gemini Pro 微调的 [AlphaCode 2]，据说这是第一个在竞技编程中达到中位竞争者水平的人工智能系统。在 Codeforces 上，在 10 次尝试内，AlphaCode 2 可以以 43% 的概率 AC，而之前的 AlphaCode 只有 25% 的概率 AC。

其主要组件包括： 
- 策略模型，为每个问题生成代码示例； 
- 采样机制，鼓励生成广泛多样的代码示例以搜索可能程序的空间； 
- 过滤机制，用于删除与问题描述不符的代码示例； 
- 聚类算法，将语义相似的代码示例分组，从而避免冗余； 
- 评分模型，用于从每个前10个代码示例集群中呈现最佳候选项。


# 参考 
- [AlphaCode 2 technical report](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf)
- [Gemini 1.0 report](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf)
- [The Best AI Model in the World: Google DeepMind’s Gemini Has Surpassed GPT-4](https://albertoromgar.medium.com/the-best-ai-model-in-the-world-google-deepminds-gemini-has-surpassed-gpt-4-1ee07f84d2ff)
- [Introducing Gemini: our largest and most capable AI model](https://blog.google/technology/ai/google-gemini-ai/#capabilities)

