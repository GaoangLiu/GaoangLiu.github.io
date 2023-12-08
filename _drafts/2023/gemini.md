---
layout:     post
title:      Gemini, The best AI model in the world 
date:       2023-12-06
tags:   [gemini, llm, deepmind]
categories: 
- nlp
---

# Google is so back

从 [ChatGPT](https://chat.openai.com) 发布之后，相当一部分用户的搜索习惯发生了改变，不再是直接在 Google 搜索，而是直接向 ChatGPT 提问，Google 搜索流量曾一度大幅下降，ChatGPT 对 Google 搜索威胁之大，据说连创始人都亲自下场搞 LLM（[Back At Google Again, Cofounder Sergey Brin Just Filed His First Code Request In Years —— Forbes](https://dlj.one/zprxug)）。其实在 AI 领域，Google 一直处于领先地位，毕竟旗下有 Google Brain 和 DeepMind 两大顶级 AI 研究机构，DeepMind 旗下的 AlphaGo 也是当年风头无两、家喻户晓的明星 AI 产品。在 NLP 领域，Google 也颇有建树，率先提出了 transformer 结构，也先后发布了 BERT、T5、ALBERT、Switch Transformer 等多个知名模型，但在聊天机器人上，对比 OpenAI 的 GPT 系列，Google 就比较滞后。在 ChatGPT 发布三个月之后，Google
才匆忙赶 [Bard](https://bard.google.com) 上架，做为 Google 的第一个 AI 聊天机器人，大家对 Bard 寄予厚望，但 Bard 效果着急太差，连基本的算术都频繁出错，语言能力更是有限，早期的版本只支持英文，对比 ChatGPT 简直是天壤之别，有人甚至调侃称为[“Bard is a joke”](https://twitter.com/high_byte/status/1639596716339896322)。为了追平 OpenAI，Google 不惜将两个 AI 研究机构合并，成立了 Google DeepMind。在今年 5 月份的 Google I/O 大会上，Google 宣布新的 Google DeepMind 实验室已开始开发 Gemini。

时隔半年，Google DeepMind 终于搞了一个大新闻，发布了 [Gemini 1.0](https://deepmind.google/technologies/gemini/) 多模态 AI 语言模型，同时支持文字、图片、音频、视频等多种输入数据的处理。根据 CEO Sundar Pichai 在 [twitter](https://twitter.com/sundarpichai/status/1732433036929589301) 上发布的 Gemini 效果展示视频中可以看到，Gemini 在音频、图片识别、文字理解、推理方面都表现出惊人的能力。此外，Gemini 在文本、音频、视频及编程等一系列开源数据集任务的评测中都刷新了 SoTA，效果非常好，可以说是目前最强综合 AI 语言模型了。

Gemini 一出，Twitter，YouTube， 微信公众号，朋友圈满屏幕都是 `Gemini`、`SoTA`、`多模态`，有点当初 ChatGPT 刚大火之后的阵仗，给人一种 Google 终于一雪前耻，一统 AI 的感觉。

<figure style="text-align: center;">
    <img src="https://image.ddot.cc/202312/gemini_openai_20231207_2247.png" width=445pt>
    <figcaption>Google is so back</figcaption>
</figure>

在学术基准测试中，Gemini 这次主要对标 GPT-4，在数学、推理、代码及综合能力等方面都（以微弱的优势）超过了 GPT-4，比如在 [MMLU](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu) 任务上。Gemini Ultra(CoT@32) 得分 90，而之前的 SoTA 由 GPT-4 (5-shots) 保持，得分为 86.4，这也是首次有模型在这个任务上得分超过人类的 89.8 分，达到了 <span style="color:blue"> 90.04 </span> 分!

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/gemini_ultra_mmlu_20231208_1116.png" width=889pt>
    <figcaption>Multi-Task Language Understanding (MMLU) on Gemini Ultra</figcaption>
</figure>

Gemini 有三个版本大小的模型，分别是：
- Gemini Ultra — 宇宙当前最强模型，适用于高度复杂的任务。
- Gemini Pro — 性能、大小和速度的最佳平衡，适用于大多数任务。
- Gemini Nano — 适合终端设备上的模型，比如在 Pixel 8 上运行。

根据 Gemini 的 [technical report](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf)，Gemini Pro 基本可以媲美 GPT-3.5，在 8 项行业标准基准测试中，有六项中表现优于 GPT-3.5，其中包括 MMLU 及 GSM8K。 Google 的官方博客 [Bard gets its biggest upgrade yet with Gemini](https://blog.google/products/bard/google-bard-try-gemini-ai/) 称 Gemini Pro 已经集成到 Bard 上，只不过当前使用的是 Gemini Pro 英文模型的微调版本，Gemini Poro 的加入增强了 Bard 的推理、规划与理解能力。Gemini Ultra 会在 2024 年初集成到 Bard Advanced 上，感觉应该就是 Google 版 ChatGPT premium 吧。 

Gemini Nano 细分还有两个版本，一个 Nano 1, 参数 1.8B，另一个 Nano 2，参数 3.25B。Google 同时还放出了基于 Gemini 的 [AlphaCode 2](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf)，据报告称，较上一代 AlphaCode 可以多解决 170% 的问题。


# Gemini Pro 使用体验 
## 视频解析能力出众，但指令对齐能力还有待提升
笔者线下感受了一下集成了 Gemini Pro 的新版 Bard，实际使用的体验是，Bard 的理解（泛化）能力还有很大的提升空间，拒绝回答问题的情况仍然普遍，尽管这个问题是它可以解决的。比如下面是 Bard 官方推荐的一条应用示例:
```
Give me insights about this video: https://www.youtube.com/watch?v=lr87yrvK86w 
Organize the information in a set of easy to scan bullet points.
```

正常情况下，Bard 会对视频进行分析，然后生成这条视频相关的一些要点。但如果我们把视频链接换成其他 YouTube 链接，比如 [Dr. Andrej Karpathy 的 State of GPT YouTube 视频](https://youtu.be/bZQun8Y4L2A?si=8D0NjPDZaLkbXW-f)，prompt 后面再加上 reply in Chinese：

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

## 计算、推理能力强，纠错能力弱
之前我们在 [Step-Back Prompting]({{site.baseurl}}/2023/12/02/Step-back-prompting/) 聊到可以通过 step-back prompting 提高 LLM 的推理能力，在 ChatGPT 上也做过测试，测试样例是一道高中物理题

| 如果温度增加 2 倍，体积增加 8 倍，理想气体的压强 $P$ 会发生什么变化？

ChatGPT 直接回答问题有一定概率出错，比如出现 Math Error，但经过提示之后可以将前面的错误改成过来，但是 Bard 就比较自信（顽固），无论如何提示，都不会改变错误的答案。上题中正确结果是“压强降为原来的 1/4”，而 Bard 一口咬定是“压强降为原来的 1/8 倍”：

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/too_confident_20231208_1109.png" width=769pt>
</figure>


# 基准测试效果
Gemini 支持文字、图片、音频、视频包括跨模型，且在每个模态上的表现都很强。Gemini 在 12 文本和推理基准测试中的 10 项，9 项图像理解基准测试，6 项视频理解基准测试，以及 5 项语音识别和语音翻译基准测试中都刷新了 SoTA。

在 [Technical Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf) 中的第一个例子里，Google 展示了 Gemini 从图片中识别学生潦草的字体，验证学生的推理过程并根据要求给出符合格式的解答。这个任务体现 了 Gemini 具备强大的图片识别、文本理解及逻辑推理能力。 

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/gemini_example_1_20231207_1505.png" width=789pt>
    <figcaption>Example 1: Gemini answers a question about a student’s handwriting</figcaption>
</figure>

一些模型的多模态是通过训练多个不同的模态模型实现，在应用中根据不同的输入选择不同的模型，而 Gemini 在训练时就是通过跨模态训练的，训练时会交叉输入文本、音频及视觉输入，后者包括图片、图表、截图、PDF 及视频，输出可以是文本或者图片。

那么一个问题是，这种联合训练的方式对比针对单一领域专门训练一个模型方法，哪一个更好。Google 称前者效果更好，因为 Gemini 一系列文本、图像、音频和视频基准测试中都取得了 SoTA。

🤔 达到了 SoTA 是不错，直观感觉也没问题，但没有使用相同数据训练一个单模态模型进行对比，这个理由未免有点牵强。


<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/crossmodal_20231208_1122.png" width=778pt>
</figure>

## 文本
<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/text_benchmark_20231208_1124.png" width=789pt>
    <figcaption>Text benchmarks</figcaption>
</figure>

如上表所示，Gemini Ultra 在几项比较重要的基准测试表现中都比较抢眼，最为突出的无疑是在 MMLU 任务上首次突破 90 分大关，比之前的 GPT-4 还要高出 3 个点。 这个突破主要利益于 Gemini Ultra 使用的一个 *uncertainty-routed chain-of-thought* 策略，这个策略在 techinical report 几句带过，没有详细解释，也没有示例。Uncertainty-rounted CoT 的原文如下：

```text
...
We proposed a new approach where model produces k chain-of-thought samples, selects the majority vote if the model is confident above a threshold, and otherwise defers to the greedy sample choice. The thresholds are optimized for each model based on their validation split performance. 
...
```

笔者暂时还没有完全理解具体是如何操作的，但看起来有种预测多个结果然后投票（consensus voting）的意思。NVIDIA 的一个 AI 工程师 [twitter@Sergio Perez](https://x.com/sergiopprz/status/1732502923022684501?s=20) 的见解类似：

```text
With the "uncertainty-routed" approach, the model generates several answers, each of them with their own CoT. If there's enough consensus among the answers, the model chooses that answer, and if not it reverts to simple maximum-likelihood sampling (i.e. no CoT) at all.
```

这个策略的效果实在是过于显著，因为 Gemini Ultra + 5-shot 效果事实上比 GPT-4 + 5-shot 低 2.7 个点，但 Gemini Ultra + 32-shot + CoT 再加上上面的 consensus voting 的方法却反超了 GPT-4 3 个点，Google 这次为了在 MMLU 上超越 GPT-4 可谓是“下足了功夫”。因此有人（e.g., [twitter@yi_ding](https://twitter.com/yi_ding/status/1732443815804653744)）表示费解的同时，也有有人对 Gemini 的训练表示了质疑。 

<figure style="text-align:center">
    <img src="https://image.ddot.cc/202312/satire_gemini_20231207_2130.png" width=389pt>
    <figcaption>The magic behind Gemini Ultra</figcaption>
</figure>   

另外一个比较显眼的结果是在 GSM8K 上达到了 94.4% 的准确率，策略是 CoT + [self-consistency](https://arxiv.org/pdf/2203.11171.pdf)，就是预测多个结果，然后投票。效果确实是好了，但这种方式计算量有点大。 

## 编程能力
Google 同时还推出基于 Gemini Pro 微调的 [AlphaCode 2](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf)，

AlphaCode 2 的主要组件包括： 
- 策略模型（policy & fine-tuning），为每个问题生成代码示例； 
- 采样机制（sampling），鼓励生成广泛多样的代码示例以搜索可能程序的空间； 
- 过滤机制（filtering），用于删除与问题描述不符的代码示例； 
- 聚类算法（clustering），将语义相似的代码示例分组，从而避免冗余； 
- 评分模型（scoring），用于从每个前10个代码示例集群中呈现最佳候选项。

报告称这是第一个在竞技编程中达到 expert 水平的人工智能系统。在 Codeforces 上测试时，给 10 次尝试机会，AlphaCode 2 有 43% 的概率 AC，而之前的 AlphaCode 只有 25% 的概率 AC。AlphaCode 2 coding 能力这么强，但为什么没有放出来让大家用用？Google 在 report 里说了：“Our system requires a lot of trial and error, and remains too costly to operate at scale. Further, it relies heavily on being able to filter out obviously bad code samples.”。简单来说，就是还不太稳定，且成本太高，仍然需要继续优化。实际上仔细观察一下上面的组件，可以看到其中在 policy 与 scoring 阶段都需要一个模型，而 AlphaCode 2 则分别微调了一个 Gemini Pro 模型，在 clustering 阶段，也需要再训练一个模型进行聚类。两个 Gemini Pro + 一个聚类模型，这个成本确实不低。

# 小结
从 2022/11/30 OpenAI 发布 ChatGPT 到现在已经刚好有一年的时间，过去的一年对 Google 来说是疲于奔命的一年。现在是 2023 的最后一个月，眼看快到圣诞节，Google 在这个时间点放出来 Gemini，估计也是急于对外展示自己仍然是 AI 领域的佼佼者。Gemini 在基准测试任务上的表现确实很好，但要实际投入使用中还有很长一段路要走，最起码现在的 Bard 还难以达到用户的要求。等到明年 Ultla 最终放出来时，OpenAI 或者 Anthropic Inc. 也许也会有新的模型发布，到时候又是一场新的竞争。


# 参考 
- AlphaCode 2 technical report: https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf
- Gemini 1.0 technical report: https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf
- Bard gets its biggest upgrade yet with Gemini: https://blog.google/products/bard/google-bard-try-gemini-ai/
- The Best AI Model in the World: Google DeepMind’s Gemini Has Surpassed GPT-4: https://albertoromgar.medium.com/the-best-ai-model-in-the-world-google-deepminds-gemini-has-surpassed-gpt-4-1ee07f84d2ff
- Introducing Gemini: our largest and most capable AI model: https://blog.google/technology/ai/google-gemini-ai/#capabilities

