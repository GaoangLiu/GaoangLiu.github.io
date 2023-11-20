---
layout:     post
title:      Low Rank Adaption
date:       2023-11-19
tags:   [lora, peft]
categories: 
- nlp
---

- [ ] LoRA 是什么，解决什么问题？
- [ ] LoRA 是如何解决这个问题的？
- [ ] 有什么特点？
- [ ] 适用场景？


由微软在 [《LoRA: Low-Rank Adaptation of Large Language Models》](https://arxiv.org/pdf/2106.09685.pdf)中提出，简单说，LoRA 是一种**降低模型可训练参数，又尽量不损失模型表现的大模型微调方法**。在 LoRA 之前，已有两类轻量化参数微调方法：Adapter 方法和 Prefix tuning 方法。



参考 
- [快速了解大模型时代爆火的lora技术](https://juejin.cn/post/7232864896744177720)