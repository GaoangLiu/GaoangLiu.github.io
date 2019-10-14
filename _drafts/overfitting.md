---
layout:     post
title:      过拟合与欠拟合
subtitle:   机器学习笔记
date:       2019-07-05
author:     ssrzz
catalog: 	true
tags:
  - machine learning
  - overfitting
---

# Overfitting

##理解过拟合

从直观表现上来说，模型过度关注于训练集本身，在训练集上表现好，但在测试集上表现不好，泛化性能差。

产生的原因可能有：
* 模型本身过于复杂，以至于拟合了训练样本集中的噪声。此时需要选用更简单的模型，或者对模型进行裁剪。
* 训练样本太少或者缺乏代表性。此时需要增加样本数，或者增加样本的多样性。
* 训练样本噪声的干扰，导致模型拟合了这些噪声，这时需要剔除噪声数据或者改用对噪声不敏感的模型。







#Reference

1. [理解过拟合, ](https://zhuanlan.zhihu.com/p/38224147)知乎专栏, 2019-03
2. 