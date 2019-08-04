---
layout:     post
title:      深度学习-VGG16
subtitle:   
date:       2019-08-04
author:     ssrzz
catalog: 	true
tags:
  - ml
  - tutorial
---



# 资源

* 文献
  * VGG-16论文题目：Very Deep Convolutional Networks for Large-Scale Image Recognition K. Simonyan, A. Zisserman 
  * VGG-16论文Arxiv链接 https://arxiv.org/abs/1409.1556
* 文件
  * VGG-16模型权重 https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz
  * TensorFlow 模型  https://www.cs.toronto.edu/~frossard/vgg16/vgg16.py

# 简介

VGG(**Visual Geometry Group**)是由K.Simonyan and A.Zisserman提出来的一个卷积神经网络模型。这个模型在ImageNet中 Top-5 测试准确率可达 92.7% （ImageNet要求从 1400 万张图片中分出1000 类)。

1. 应用 
   * 给定图片 —> 找到对应图片的分类
   * 可适用于1000类图片
   * 图片输入尺寸 224 * 224 * 3
2. 结构
   * 卷积层
   * 最大池化层
   * 全连接层
   * 总计16层
     * Convolution using 64 filters
     * Convolution using 64 filters + Max pooling
     * Convolution using 128 filters
     * Convolution using 128 filters + Max pooling
     * Convolution using 256 filters
     * Convolution using 256 filters
     * Convolution using 256 filters + Max pooling
     * Convolution using 512 filters
     * Convolution using 512 filters
     * Convolution using 512 filters + Max pooling
     * Convolution using 512 filters
     * Convolution using 512 filters
     * Convolution using 512 filters + Max pooling
     * Fully connected with 4096 nodes
     * Fully connected with 4096 nodes
     * Output layer with Softmax activation with 1000 nodes
3. 属性
   * 模型大小  528MB
   * Top-1 Accuracy: 70.5%
   * Top-5 Accuracy: 90.0%
4. 预训练模型 [pre trained model](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz)