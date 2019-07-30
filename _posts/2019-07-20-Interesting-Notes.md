---
layout:     post
title:      Interesting Notes
subtitle:   一些小知识点+碎碎念
date:       2019-07-20
author:     ssrzz
catalog: 	true
tags:
  - AI
  - ML
---



* TPU 果然比 GPU快太多了 

* Ml 的根本问题是**优化与泛化之间的对立** 。 优化：调节模型以在训练数据上得到最佳性能； 泛化：训练好的模型在前所未见的数据上的性能好坏。 理想的模型是刚好在欠拟合和过拟合的界线上，在容量不足和容量过大的界线上。为了找到这条界线，你必须穿过它。 
  
* ML最终目的是得到良好的泛化，但你无法控制泛化，只能基于训练数据调节模型。
* you are stronger than you thought.
* ffmpeg -i input.mkv -vf fps=5 face%04d.jpg -hide_banner 从input.mkv中抽取视频，每秒抽取5个
