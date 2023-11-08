---
layout:     post
title:      Background propagation formula derivation
date:       2023-01-02
tags: [nlp, gradient, bp]
categories: 
- nlp
---

反向传播（Backpropagation，缩写为BP）是“误差反向传播”的简称，是一种与最优化方法（e.g., 梯度下降法）结合使用的，用来训练人工神经网络的常见方法。该方法对网络中所有权重计算损失函数的梯度。这个梯度会反馈给最优化方法，用来更新权值以最小化损失函数。 

BP 在前向传播过程计算（并缓存）每个节点的输出值，然后再按反向传播遍历图的方式计算损失函数值相对于每个参数的偏导数。

用 $\theta$ 表示模型要学习的参数，$J$ 表示损失函数，$\eta$ 表示学习率，BP 的更新公式为：

$$\theta = \theta - \eta \frac{\partial J}{\partial \theta}$$

BP 求解的是 $\frac{\partial J}{\partial \theta}$，<span style="color:blue;">即损失函数相对于参数的偏导数</span>。这里的 $\frac{\partial J}{\partial \theta}$ 是一个向量，它的每个元素是 $\frac{\partial J}{\partial \theta_i}$，即损失函数相对于参数 $\theta_i$ 的偏导数。


考虑一个简单的三层网络，如下图所示：

<div style="display: flex; justify-content: center; align-items: center;">
<img src="https://file.ddot.cc/imagehost/2023/3_layer.png" width="678px">
</div>

假定：
- 输入层有 $n$ 个神经元，输出层有 $m$ 个神经元，隐藏层有 $h$ 个神经元。
- $x_i^1$ 表示输入的第 $i$ 个特征，$x^1$ 表示输入层的输入，$x^1 = [x_1^1, x_2^1, \cdots, x_n^1]$。
- $y_i$ 表示输出层的第 $i$ 个输出，$y = [y_1, y_2, \cdots, y_m]$。
- $w_{jk}^l$ 表示第 $l-1$ 层的第 $k$ 个神经元到第 $l$ 层的第 $j$ 个神经元的权重。
- $b_j^l$ 表示第 $l$ 层的第 $j$ 个神经元的偏置。
- $z_j^l$ 表示第 $l$ 层的第 $j$ 个神经元的输入，$z_j^l = \sum_{k=1}^{n} w_{jk}^l a_k^{l-1} + b_j^l$。
- $a_j^l$ 表示第 $l$ 层的第 $j$ 个神经元的输出，$a_j^l = \sigma(z_j^l)$，其中 $\sigma$ 表示激活函数。

为方便起见，用均值平方误差（Mean Squared Error，MSE）作为损失函数，即： $J = \frac{1}{2 m} \sum_{i=1}^{m} (y_i - a_i^l)^2$，用 Sigmod 作为激活函数。


首先，前向传播的计算方式如下：

$$ \begin{aligned}
z_j^l &= \sum_{k=1}^{n} w_{jk}^l a_k^{l-1} + b_j^l \\
a_j^l &= \sigma(z_j^l)
\end{aligned} $$

直觉：获得最终模型的输出结果 $a_j^l$ 之后，它跟真实结果的误差为 $y_j - a_j^l$，损失函数为 $J = \frac{1}{2  m} \sum_{i=1}^{m}(y_i - a_i^l)^2$。理想状态下，模型完美拟合，误差为 0，即 $J = 0$。但是，实际上模型的输出结果 $a_j^l$ 与真实结果 $y_j$ 之间的误差是不可避免的，这时我们可以利用损失来指导模型的学习，即通过调整模型的参数，使得损失函数 $J$ 最小化。
四大步骤：
1. 计算损失函数 $J$ 相对于输出层的偏导数 $\frac{\partial J}{\partial a_j^l}$ (上标 $l$ 指第 $l$ 层网络）。
2. 计算损失函数 $J$ 相对于输出层的输入 $z_j^l$ 的偏导数 $\frac{\partial J}{\partial z_j^l}$。
3. 计算损失函数 $J$ 相对于输出层的权重 $w_{jk}^l$ 的偏导数 $\frac{\partial J}{\partial w_{jk}^l}$。
4. 计算损失函数 $J$ 相对于输出层的偏置 $b_j^l$ 的偏导数 $\frac{\partial J}{\partial b_j^l}$

第一步，计算损失函数 $J$ 相对于<span style="color:blue;"> 输出层</span>的偏导数 $\frac{\partial J}{\partial a_j^l}$：
$$ \begin{aligned}
\frac{\partial J}{\partial a_j^l} &= \frac{\partial}{\partial a_j^l} \frac{1}{2 m} \sum_{i=1}^{m} (y_i - a_i^l)^2 \\
&= \frac{1}{2 m} \sum_{i=1}^{m} \frac{\partial}{\partial a_j^l} (y_i - a_i^l)^2 \\
&= \frac{1}{2 m} \sum_{i=1}^{m} 2 (y_i - a_i^l) \frac{\partial}{\partial a_j^l} (y_i - a_i^l) \\
&= \frac{1}{m} \sum_{i=1}^{m} (y_i - a_i^l) \frac{\partial}{\partial a_j^l} (- a_i^l) \\
&= \frac{1}{m} \sum_{i=1}^{m} (y_i - a_i^l) (-1) \\
&= \frac{1}{m} \sum_{i=1}^{m} (a_i^l - y_i)
\end{aligned} $$
写成向量形式，即： $\frac{\partial J}{\partial a^l} = \frac{1}{m} (a^l - y)$，其中 $a^l = [a_1^l, a_2^l, \cdots, a_m^l]$，$y =[y_1, y_2, \cdots, y_m]$。
计算这一层的意义：当输出层的输出 $a_j^l$ 增加 1 个单位时，损失函数 $J$ 的变化量是 $\frac{\partial J}{\partial a_j^l}$。


第二步，计算损失函数 $J$ 相对于<span style="color:blue;"> 输出层的输入 </span> $z_j^l$ 的偏导数 $\frac{\partial J}{\partial z_j^l}$：
$$ \begin{aligned}
\frac{\partial J}{\partial z_j^l} &= \frac{\partial J}{\partial a_j^l} \frac{\partial a_j^l}{\partial z_j^l} \\
&= \frac{1}{m} \sum_{i=1}^{m} (a_i^l - y_i) \frac{\partial}{\partial z_j^l} \sigma(z_j^l) \\
&= \frac{1}{m} \sum_{i=1}^{m} (a_i^l - y_i) \sigma'(z_j^l) \\
&= \frac{1}{m} \sum_{i=1}^{m} (a_i^l - y_i) \sigma(z_j^l) (1 - \sigma(z_j^l)) \\
&= \frac{1}{m} \sum_{i=1}^{m} (a_i^l - y_i) a_j^l (1 - a_j^l)
\end{aligned} $$

第三步，计算损失函数 $J$ 相对于 <span style="color:blue;"> 输出层的权重 </span> $w_{jk}^l$ 的偏导数 $\frac{\partial J}{\partial w_{jk}^l}$：
$$ \begin{aligned}
\frac{\partial J}{\partial w_{jk}^l} &= \frac{\partial J}{\partial z_j^l} \frac{\partial z_j^l}{\partial w_{jk}^l} \\
&= \frac{1}{m} \sum_{i=1}^{m} (a_i^l - y_i) a_j^l (1 - a_j^l) \frac{\partial}{\partial w_{jk}^l} \left( \sum_{k=1}^{n} w_{jk}^l a_k^{l-1} + b_j^l \right) \\
&= \frac{1}{m} \sum_{i=1}^{m} (a_i^l - y_i) a_j^l (1 - a_j^l) a_k^{l-1} \\
\end{aligned} $$


第四步，计算损失函数 $J$ 相对于 <span style="color:blue;"> 输出层的偏置 </span> $b_j^l$ 的偏导数 $\frac{\partial J}{\partial b_j^l}$：
$$ \begin{aligned}
\frac{\partial J}{\partial b_j^l} &= \frac{\partial J}{\partial z_j^l} \frac{\partial z_j^l}{\partial b_j^l} \\
&= \frac{1}{m} \sum_{i=1}^{m} (a_i^l - y_i) a_j^l (1 - a_j^l) \frac{\partial}{\partial b_j^l} \left( \sum_{k=1}^{n} w_{jk}^l a_k^{l-1} + b_j^l \right) \\
&= \frac{1}{m} \sum_{i=1}^{m} (a_i^l - y_i) a_j^l (1 - a_j^l) \\
\end{aligned} $$


真正需要更新的是 $w_{jk}^l$ 和 $b_j^l$，前两项的计算是为了计算后两项。我们可以将上面的公式写成向量形式：
$$ \begin{aligned}
\frac{\partial J}{\partial w^l} &= \frac{1}{m} (a^{l-1})^T (a^l - y) \odot \sigma'(z^l) \\
\frac{\partial J}{\partial b^l} &= \frac{1}{m} \sum_{i=1}^{m} (a_i^l - y_i) \odot \sigma'(z^l)
\end{aligned} $$

其中，$\odot$ 表示向量的逐元素相乘，$\sigma'(z^l)$ 表示向量 $z^l$ 中每个元素的激活函数的导数。

最后，我们可以得到 BP 的更新公式：

$$ \begin{aligned}
w^l &= w^l - \eta \frac{\partial J}{\partial w^l} \\
b^l &= b^l - \eta \frac{\partial J}{\partial b^l}
\end{aligned} $$

其中，$\eta$ 表示学习率。
