---
layout: post
title: Prefix Tunning
date: 2023-11-19
tags: prefix-tunning peft
categories: nlp
author: berrysleaf
---
* content
{:toc}


- [ ] Prefix-tunning 是什么，解决什么问题？



- [ ] 是如何解决这个问题的？
- [ ] 有什么特点？
- [ ] 适用场景？


主流的 NLP 任务都是 pretraining + fine-tuning 的范式，即在预训练模型的基础上，针对特定任务进行微调。这种方法的优点是简单，但在当下模型越来越大的情况下，fine-tuning 的成本也越来越高。另外，fine-tuning 也有一些缺点，例如，模型的泛化能力不强，对于一些小数据集，模型的效果很差。针对这些问题，有一些研究者提出了一些方法，例如，[《Prefix-Tuning: Optimizing Continuous Prompts for Generation》](https://arxiv.org/pdf/2101.00190.pdf) 就是一种新的 fine-tuning 方法，它可以在不改变模型参数的情况下，通过修改输入的前缀来优化模型的效果。这种方法的优点是可以在不改变模型参数的情况下，优化模型的效果，而且可以在小数据集上取得很好的效果。

是一种轻量的 NLG 任务调参技术，解决 fine-tuning 方法在 LLM 上