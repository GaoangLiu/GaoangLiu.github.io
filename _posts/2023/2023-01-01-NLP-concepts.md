---
layout: post
title: NLP-concepts
date: 2023-01-01
tags: nlp
categories: nlp
author: berrysleaf
---
* content
{:toc}


## Autoregressive models

AR 是基于**目标变量历史数据的组合对目标变量进行预测**的模型，自回归一词中的自字即表明其是对变量自身进行的回归。




一个 $$p$$ 阶的自回归模型可以表示为：

$$y_t= c + \phi_1 y_{t-1} + ... + \phi_p y_{t-p} + \epsilon_t$$

这里的 $$\epsilon_t$$ 是白噪声，$$c$$ 是常数，$$\phi_1, ..., \phi_p$$ 是参数，$$p$$ 是滞后阶数。

NLP 中提到 AR 一般指自回归语言模型，比如 GPT。AR 的核心思想是，利用前面的词语序列，预测下一个词语的概率分布，从而生成连续的文本。

## Beam search
贪心搜索是一种启发式搜索算法，它在每一步都选择最优的选择，从而得到全局最优解。贪心搜索的问题在于，它只考虑了当前步骤的最优解，而没有考虑到后续步骤的最优解，因此可能会导致局部最优解。
穷举搜索是一种暴力搜索算法，它会遍历所有可能的解，从而得到全局最优解。穷举搜索的问题在于，它的时间复杂度是指数级的，因此在实际应用中很少使用。
Beam search 是一种折中的方法，它在每一步都保留 $$k$$ 个最优解，相对 greedy 搜索增大了搜索空间。

## 激活函数 

可以向深度学习网络中引入非线性因素，旨在帮助网络学习到数据中的复杂模型。

| 激活函数 | 公式 |
| --- | --- | 
| sigmoid | $$\sigma(x) = \frac{1}{1 + e^{-x}}$$ |
| tanh | $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$ |
| ReLU | $$\text{ReLU}(x) = \max(0, x)$$ |
| Leaky ReLU | $$\text{LeakyReLU}(x) = \max(0.01x, x)$$ | 
| PReLU | $$\text{PReLU}(x) = \begin{cases} x & x \geq 0 \\ \alpha x & x < 0 \end{cases}$$ |
| ELU | $$\text{ELU}(x) = \begin{cases} x & x \geq 0 \\ \alpha (e^x - 1) & x < 0 \end{cases}$$ | 
| SELU | $$\text{SELU}(x) = \lambda \begin{cases} x & x \geq 0 \\ \alpha (e^x - 1) & x < 0 \end{cases}$$ | 
| Swish | $$\text{Swish}(x) = x \cdot \sigma(x)$$ |
| [GELU](#activation-gelu) | $$\text{GELU}(x) = x \cdot \Phi(x)$$ , $$\Phi(x)$$ 是高斯分布的累积分布函数 |

### Sigmoid
<img src='https://upload.wikimedia.org/wikipedia/commons/5/5b/Activation_logistic.svg' width=345pt>

饱和激活函数。Sigmoid 函数的输出范围为 $$(0, 1)$$，可以用作二分类问题的输出层。缺点是，容易出现梯度消失问题。因为 Sigmoid 函数的导数为 $$\sigma^{\prime}(x) = \sigma(x)(1 - \sigma(x))$$，当 $$x$$ 的绝对值较大时，$$\sigma(x)$$ 的值会接近于 0 或 1，从而导致梯度接近于 0，即梯度消失问题，导致模型收敛缓慢。

由于这个原因，现在不鼓励使用 Sigmoid 函数，当必须要使用 Sigmoid 函数时，Tanh 函数通过比 Sigmoid 表现更好。



### Tanh
双曲正切函数，与 Sigmoid 紧密相关，$$g(x) = 2\sigma(2x) - 1$$。

<img src='https://upload.wikimedia.org/wikipedia/commons/8/87/Hyperbolic_Tangent.svg' width=345pt>

饱和激活函数。Sigmoid 函数不是关于原点中心对称的（zero-centered），Tanh 解决了这个问题。 但跟 Sigmoid 一样，Tanh 也容易出现梯度消失问题。

### ReLU
<img src='https://upload.wikimedia.org/wikipedia/commons/6/6c/Rectifier_and_softplus_functions.svg' width=345pt>

非饱和激活函数。ReLU 函数的输出范围为 $$[0, +\infty)$$，计算简单，只需要判断输入是否大于 0。ReLU 有非常好的特性：
1. 单侧抑制。 当输入小于 0 时，ReLU 函数的输出为 0，神经元处理抑制状态，从而减少了模型的计算量。
2. 激活边界宽阔。 当输入大于 0 时，神经元全部出于激活状态，激活值取值边界无穷大，无饱和区。
 - Sigmoid 存在饱和区，当输入较大或较小时，梯度接近于 0，更新缓慢，导致模型收敛缓慢。
3. 稀疏性。 Sigmoid 把抑制区设置为一个极小值，但是不为 0，因此要参与运算，而 ReLU 的抑制区的结果直接为 0，不参与后续计算，简单粗暴的造成网络稀疏性，而且计算十分简单。

ReLU 凭借上述优点在深度学习中得到了广泛的应用，但是也存在一些问题：
1. Dead ReLU 问题：当 ReLU 的输入小于零时，梯度变为零，导致神经元无法更新权重，从而导致神经元对输入不再敏感，被称为"死亡神经元"。
2. 输出不是中心化：ReLU 的输出范围在 $$[0, +\infty)$$ 之间，不是以 0 为中心的，这会导致后续层的输入分布发生偏移，从而影响模型的收敛速度。
3. 非线性分割能力有限：ReLU 只考虑输入是否大于零，而不关心不同的非线性级别，这可能导致模型对于复杂数据集的拟合能力有限。


### Leaky ReLU
<img src='https://upload.wikimedia.org/wikipedia/commons/a/ae/Activation_prelu.svg' width=345pt>

Leaky ReLU 是对 ReLU 的改进，当输入小于 0 时，Leaky ReLU 的输出为 $$\alpha x$$，其中 $$\alpha$$ 是一个小于 1 的超参数，通常取 0.01。Leaky ReLU 的优点是，当输入小于 0 时，梯度不为 0，可以缓解神经元死亡问题。缺点与 ReLU 相同，输出不是以 0 为中心的，这会导致后续层的输入分布发生偏移，从而影响模型的收敛速度。


### ELU
<img src='https://image.ddot.cc/202311/elu_rc.png' width=345pt>

$$\text{ELU}(x) = \begin{cases} x & x \geq 0 \\ \alpha (e^x - 1) & x < 0 \end{cases}$$

ELU 是对 ReLU 的改进，当输入小于 0 时，ELU 的输出为 $$\alpha (e^x - 1)$$，其中 $$\alpha$$ 是一个超参数，通常取 1。
ELU 函数在原点附近具有**平滑的斜率**，有助于梯度的稳定传播。相比于 ReLU 等激活函数，在原点附近避免了梯度的不连续性。此外，与 ReLU 函数不同，ELU 函数对负值输出非零值，从而保留了一定的负值信息。这可以更好地处理负值输入，避免了神经元死亡问题。
ELU 函数在实际应用中的效果通常优于其他激活函数，并且具有更好的梯度流动性和稳定性。但由于包含了指数计算，ELU 函数的计算量较大。


### GELU  <span id='activation-gelu'></span>
<img src='https://miro.medium.com/v2/resize:fit:1100/format:webp/1*kwHcbpKUNLda8tvCiwudqQ.png' width=345pt>

$$\text{GELU}(x) = x \cdot P(X \leq x) = x \cdot \Phi(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$$


GELU(Gaussian Error Linear Units，高斯误差线性单元) 是对 ReLU 的改进，当输入小于 0 时，GELU 的输出为 $$x \cdot \Phi(x)$$，其中 $$\Phi(x)$$ 是高斯分布的累积分布函数。GELU 函数在原点附近具有平滑的斜率，有助于梯度的稳定传播。相比于 ReLU 等激活函数，在原点附近避免了梯度的不连续性。


### 神经元死亡 

神经元死亡是指神经元的输出恒为 0，从而导致该神经元无法对输入进行学习。成因，

$$w' = w - \eta \Delta w$$

当学习率 $$\eta$$ 过大，导致 $$\Delta w$$ 过大，更新后的 $$w'$$ 为负值，进而导致神经元输出恒为 0。后续的反向传播过程中，该神经元的梯度恒为 0，权重无法更新。


缓解方案:
1. 使用 LeakyReLU、Parametric ReLU 或 ELU 等改进版的激活函数；
2. 减小学习率； 
3. 采用 momentum 或者 Adam 等自适应优化算法；


## 数据白化 



# 量化

PyTorch 对量化的支持：

1. 训练感知量化（Quantization Aware Training, QAS）。
2. 训练结束后的动态量化（Post Training Dynamic Quantization）。
3. 训练结束后的静态量化（Post Training Static Quantization）。

2 与 3 的区别在于，2 是在模型的输入输出上进行量化，3 是在模型的权重上进行量化。


# Loss functions

## Sparse softmax <span id='sparse-softmax'> </span>

用于需要输出一个 sparse output 的场景，比如推荐系统中，只保留一些最相关的选项。原作[《From Softmax to Sparsemax:A Sparse Model of Attention and Multi-Label Classification》](https://arxiv.org/pdf/1602.02068.pdf)称可增强解释性并提升效果。 参考苏老师设计一个[版本](https://spaces.ac.cn/archives/8046/comment-page-2):

||orginal|sparse|
|---|---|---|
|softmax|$$p_i=\frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}$$|$$p_i=\begin{cases}\frac{e^{x_i}}{\sum_{j \in \Omega_k} e^{x_j}}& i \in \Omega_k \\0& i \not \in \Omega_k\end{cases}$$|

其中 $$\Omega_k$$ 是将 $$x_1, x_2, ..., x_n$$ 从大到小排序后的前 $$k$$ 个元素的下标集合。思路是，计算出来结果后，只保留前 $$k$$ 个最大的概率，其余的概率置为 0。$$k$$ 是一个超参数，$$k=n$$时，等价于原始的 softmax。

Torch 版本的一个实现参考[Github](https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py)。

