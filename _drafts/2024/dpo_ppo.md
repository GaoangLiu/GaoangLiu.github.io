---
layout:     draft
title:      DPO and PPO
date:       2024-01-10
tags: [rlhf,dpo,ppo]
categories: 
- nlp
---



# 直接偏好优化
what, why, how?

直接偏好优化（Direct Perference Optimization,DPO）是一种 LM 偏好对齐算法，最初 Rafailov 等人在 [《Direct Preference Optimization: Your Language Model is Secretly a Reward Model》](https://arxiv.org/abs/2305.18290) 中提出。

在此之前，让LM对人类偏好对齐常用的算法是RLHF，思路是先根据人类偏好拟合一个奖励模型，再用强化学习的方法去微调一个LM，使得LM的输出尽可能奖励最大化。但RLHF复杂，且不稳定。

优点是什么？
稳定、效果好、计算量小。

先说稳定。 

## RLHF 的三个阶段 
第一个阶段是**有监督微调（superivsed fine-tuning, STF）**：使用高质量数据集对 LM 进行有监督微调，输出模型 $\pi^\text{SFT}$。


第二阶段是奖励模型训练（reward modeling），给定一个 prompt $x$, 用 $\pi^\text{SFT}(y \vert x)$ 输出两个序列 $y_1$ 和 $y_2$，然后人类选择更好的那个序列 $y_w \succ y_l$，这样就可以得到一个数据集 $\mathcal{D}=\{(x, y_w, y_l)\}$。假设这个倾向数据集是由一个 latent reward model $r_\theta(y,x)$ 生成的，那么就可以用最大似然估计（MLE）来训练奖励模型，使得 $r_\theta(y_w,x) > r_\theta(y_l,x)$。

Bradley-Terry 模型是一种常用的偏好模型，它假设人类偏好分布$p(y_1 \succ y_2 \vert x)$可以由 $r_\theta(y, x)$ 生成，即 $p(y_1 \succ y_2 \vert x) = \frac{\exp(r_\theta(y_1,x))}{\exp(r_\theta(y_1,x)) + \exp(r_\theta(y_2,x))}$。那么最大似然估计就是 $\theta^* = \arg\max_\theta \sum_{(x, y_w, y_l) \in \mathcal{D}} \log \frac{\exp(r_\theta(y_w,x))}{\exp(r_\theta(y_w,x)) + \exp(r_\theta(y_l,x))}$，其中 $D$ 是倾向数据集。

最终得到的 $r_\theta^*$ 就是一个奖励模型，它可以用来评估一个序列的好坏。

第三阶段是**强化学习微调（RL fine-tuning, RLFT）**，使用 $r_\theta^*$ 作为奖励函数，对 $\pi^\text{SFT}$ 进行强化学习微调，得到最终的模型 $\pi^\theta$。等同于以下优化问题：

$$\max_{\pi^\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi^\theta(y\vert x)}[r_\theta^*(y,x)] - \beta \cdot \mathbb{D}_{KL}(\pi^\theta(y \vert x)  \vert  \vert  \pi^\text{SFT}(y \vert x))$$

其中 $\beta$ 是一个超参数，用来平衡两个目标。

