---
layout: post
title: Normalization
date: 2023-10-11
tags: normalization
categories: nlp
author: gaoangliu
---
* content
{:toc}


# Batch Normalization

[Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf)，常简称 BatchNorm 或者 BN，是由 Google 的 Sergey Ioffe 等于 2015 年提出的一种数据归一化方法，主要解决在深度学习中遇到的 ICS（Internal Covariate Shift） 问题。通常用在网络激活层之前，可以加快模型训练时的收敛速度，使得模型训练过程更加稳定，避免梯度爆炸或者梯度消失。并且起到一定的正则化作用，几乎代替了Dropout。





BN 是逐个特征进行归一化的，从向量的角度来看，在一个 batch 内，是对每个维度分别进行归一化。 形式化地，令 $$B$$ 表示一个 batch，大小为 $$m$$，那么 $$B$$ 的经验均值及方差为：

$$\mu_B = \frac{1}{m} \sum_{i=1}^m {x_i}, \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m {(x_i - \mu_B)^2}$$

假设深度网络第 $$l$$ 层的输入为 $$d$$ 维，即样本 $$x$$ 形如 $$x = (x^1, x^2, \cdots, x^d)$$，那么逐维归一化的输出为：

$$x_i^{\prime j} = (x_i^j - \mu_B^j) / \sqrt{(\sigma_B^{j})^2 + \epsilon}$$

其中 $$j$$ 表示第 $$j$$ 维，$$\epsilon$$ 是一个很小的数，防止分母为 0。$$\mu_B^j$$ 和 $$\sigma_B^j$$ 分别表示第 $$j$$ 维的均值和方差。归一化后的输出 $$x_i^{\prime j}$$ 的均值和方差为 0 和 1。

最后，对归一化后的输出进行线性变换：

$$y_i^j = \gamma^j x_i^{\prime j} + \beta^j$$

其中 $$\gamma^j$$ 和 $$\beta^j$$ 是可学习的参数。

总结下来，流程是 1. 每个 batch 求均值与方差； 2. 对每个样本的每个特征进行归一化； 3. 对归一化后的结果进行线性变换（缩放及平移）。

## 工作机制

[BatchNorm](https://arxiv.org/abs/1502.03167) 的作者认为它可以有效缓解内协变量转移（internal covariate shift）的问题，即参数初始化和每个层输入分布的变化会影响网络的学习速度。也有[工作](https://arxiv.org/abs/1805.11604)认为批标准化并不会减少内协变量转移，而是使目标函数平滑，从而提高性能。[还有工作](https://arxiv.org/abs/1805.10694)认为，批标准化实现了长度和方向的解耦，从而加速神经网络的训练。

关于 Internal Covariate Shift（ICS），BN 论文作者给出的表述性定义：在深层网络训练的过程中，由于**网络中参数变化而引起内部结点数据分布发生变化的这一过程**被称作 Internal Covariate Shift。ICS 带的问题是，在通过激活层之后，激活值容易陷入到激活层的梯度饱和区，这种现象会发生在对模型应用饱和激活函数时，比如 sigmoid, tanh 等，这会导致梯度消失，从而降低模型收敛速度。 

在 BN 之前，缓解 ICS 也有一些方案，比如：
1. 采用非饱和激活函数，比如 Relu，Elu 等。
2. 使用更小的学习率。
3. 使用更好的参数初始化方法，比如 Xavier 初始化，He 初始化等。
4. 数据白化 （e.g., [《A convergence analysis of log-linear training》](https://papers.nips.cc/paper_files/paper/2011/hash/e836d813fd184325132fca8edcdfb40e-Abstract.html)）。 

其中，白化是在模型的每一层输入上，采用一种线性变化（例如 PCA），使得输入的特征具有相同的均值和方差。例如采用 PCA，就让所有特征的分布均值为0，方差为1去除特征之间的相关性。

然而，天下没有免费的午餐，在每一层使用白化，给模型增加了运算量。而小心地调整学习速率或其他参数，又陷入到了超参调整策略的复杂中。因此，BN 作为一种更优雅的解决办法应运而生。 

BN 作者声称，BN 可以缓解内协变量转移的问题，因为批标准化使得每一层的输入分布都是固定的，这样就不会因为参数的变化而导致输入分布的变化，从而加速网络的训练。

但后续的工作对这一点提出了质疑，比如[《How Does Batch Normalization Help Optimization?》](https://arxiv.org/abs/1805.11604) 。作者做了一个实验，使用了三种不同的训练模式来训练 VGG-16 网络：标准模式（无批量归一化），批量归一化以及在训练过程中向每层添加噪声的批量归一化。在第三个模型中，噪声具有非零均值和非单位方差，即显式引入了协变量漂移。但最终模型的准确性与第二个模型相当，并且都比第一个模型表现更好，这表明协变量漂移不是批量归一化改善性能的原因。

### 关于损失平滑
[《How Does Batch Normalization Help Optimization?》](https://arxiv.org/abs/1805.11604) 作者认为 BatchNorm 的真正作用是**将底层优化问题重新参数化，使其曲面更加平滑**（it reparametrizes the underlying optimization problem to make its landscape significantly more smooth）。直接影响是损失函数更加 Lipschitz 化（？），损失变化及梯度的幅度都会减小。相比之下，未使用 BatchNorm 的深度网络中，损失函数不仅是非凸的，而且往往具有大量的“皱褶”，平坦区域和区域陡峭的最小值。在进行 BN 操作之后，梯度更可靠且预测性更强，训练过程中不用特别担心梯度爆炸或者梯度消失的问题，这意味着我们可使用更大的学习率，且模型对超参数的选择更加鲁棒。

<img src="https://image.ddot.cc/202311/gradients_change_rc.png" width=567pt>

作者对比了 w/ BatchNorm 和 w/o BatchNorm 的梯度变化情况，可以看到，使用 BatchNorm 后，梯度变化幅度更小，更加平滑。

一句话总结而言，BN 平滑了 loss，保持了梯度下降时稳定。 

> note, $$f$$ 是 L-Lipschiz 的，如果对  $$\forall x_1, x_2, s.t., \lvert f(x_1) - f(x_2) \rvert \leq L \lVert x_1 - x_2 \rVert $$

## 推理 
训练及推理都需要 BatchNorm，差别在于推理时不在 batch 内计算均值与方差，因为很可能只有一个样本。因此，在训练过程中，同时计算滑动平均(moving average)均值与方差，在推理阶段直接使用。
```
running_mean = momentum * running_mean + (1 - momentum) * sample_mean
running_var = momentum * running_var + (1 - momentum) * sample_var
```

## Pros & Cons
BatchNorm 有以下几个优点：
- 加速收敛：Batch Normalization 使得网络中间层的输入值更加稳定，从而加速网络训练。
- 正则化：Batch Normalization 有一定的正则化效果，可以减少 Dropout 等其他正则化手段的使用。
- 允许使用较大的学习率：Batch Normalization 使得网络中间层的输入值更加稳定，可以允许使用较大的学习率，加快网络训练速度。


局限性：
- 首先是<span style="color: blue;">计算量有所增加</span>，因为需要在每一层计算 batch 的均值与方差，同时还增加了参数量。
- 其次，效果对 <span style="color: blue"> batch size 敏感 </span>，如果 batch size 太小，则不具备代表性，添加 BN 反而效果更差。

## [Python 实现]
参考参考[Github](https://github.com/gaoangliu/gaoangliu.github.io/blob/master/codes/2023/batchnorm.py)的 Python 实现。



# Layer Normalization

[Layer Normalization](https://arxiv.org/abs/1607.06450) 是 Batch Normalization 的一个变种，计算公式如下：

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} $$
$$y = \gamma \hat{x} + \beta$$

其中，$$\mu$$ 和 $$\sigma$$ 分别是输入 $$x$$ 的均值和标准差，$$\epsilon$$ 是一个小的常数用于防止除以零的情况，$$\gamma$$ 和 $$\beta$$ 是可学习的缩放因子和偏移因子。

Batch Normalization 的处理对象是对一批样本， Layer Normalization 的处理对象是单个样本。Batch Normalization 是对这批样本的同一维度特征做归一化， Layer Normalization 是对这单个样本的所有维度特征做归一化。

Layer Normalization 有以下几个优点：

- 不依赖于 batch size：Batch Normalization 是对每个 mini-batch 的数据进行归一化，因此 batch size 的大小会影响归一化的效果。而 Layer Normalization 是对每个样本的所有特征进行归一化，因此不依赖于 batch size 的大小。
- 适用于 RNN：Batch Normalization 依赖于 batch size 的大小，因此不适用于 RNN。而 Layer Normalization 不依赖于 batch size 的大小，因此适用于 RNN。

# 二者对比 

BN、LN 可以看作横向和纵向的区别。经过归一化再输入激活函数，得到的值大部分会落入非线性函数的线性区，导数远离导数饱和区，避免了梯度消失，这样来加速训练收敛过程。

- 方式：BN 是对一个 batch-size 样本内的每个特征做归一化。LN 是对每个样本的所有特征做归一化。
- 范围：BN 的转换是针对单个神经元可训练的：不同神经元的输入经过再平移和再缩放后分布在不同的区间；而 LN 对于一整层的神经元训练得到同一个转换：所有的输入都在同一个区间范围内。
- 局限性：如果不同输入特征不属于相似的类别（比如颜色和大小），那么 LN 的处理可能会降低模型的表达能力。


## Normalization V.S. Standardization

额外提一点，虽然 normalization 经常和 standardization 混合使用，中文里*标准化*与*归一化*也经常混着用，但二者还是有一些差异的。Normalization 是将数据缩放到 [0, 1] 区间，Standardization 是将数据缩放到均值为 0，方差为 1 的标准正态分布。Normalization 适用于数据分布有明显边界的情况，比如图像数据，其像素值范围为 [0, 255]；Standardization 适用于数据分布没有明显边界的情况，比如身高、体重等数据。


# FAQ
## 归一化的意义？

归一化的目地是，将数据规整到统一区间，减少异常值的影响，使得数据更加稳定，有利于模型的训练。

## 为什么 BN 在归一化后，还要逐个特征进行线性变换？即使用 $$\gamma$$, $$\beta$$ 进行还原？

主要原因是，保证模型的表征能力。BatchNorm 将数据的分布调整到均值为 0、方差为 1 的标准正态分布，这样可以避免梯度消失和梯度爆炸问题。但有时候，标准正态分布并不是最适合当前任务的数据分布。比如 BatchNorm 后的输出 $$y \propto \mathcal{N}(0, 1)$$, 根据 $$3\sigma$$ 法则，多数 y 值都将位于 [-2, 2] 之间。我们对 $$y$$ 随机采样，再经过激活函数 sigmoid。如下图所见红色点集所示，激活值也都聚集在 sigmoid 函数非饱和区域。这个区域可以用一个线性函数近似，因此模型的表征能力会受到限制。

<img src="https://image.ddot.cc/202311/normalization_sigmoid_rc.png" width=567pt>

另外一点，缩放和平移系数 $$\gamma, \beta$$ 没有写死，而是设定成**可学习的**，这样是为了增加模型的灵活性。模型可以自动学习适当的参数来调整归一化后的数据，从而能够自适应地学习数据的特征。


