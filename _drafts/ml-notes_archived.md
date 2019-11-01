---
layout:     post
title:      Machine Learning Notes
date:       2019-06-25
tags: [machine learning]
categories:
  - note
---

<h1 style="font-size:bold;">🌵Basic Math
 </h1> 

最小二乘法（英语：least squares method），又称最小平方法，是一种数学优化方法。它通过最小化误差的平方和寻找数据的最佳函数匹配。
<h2 style="font-size:bold; font-family: STHeiti, SimHei, STKaiti, KaiTi">🌲🌲 张量
 </h2> 
标量：0阶张量

向量：1阶张量


<h1 style="font-size:bold;">🌵Machine Learning Basics
 </h1> 
<h2 style="font-size:bold; font-family: STHeiti, SimHei, STKaiti, KaiTi">🌲🌲  Concepts
 </h2> 
<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ 什么是机器学习
 </h3> 

- Arthur Samule(1959): fields of study that gives computers ability to learn without being explicitly programmed. 

* Tom Mitchell(1998): A computer program is said to learn some experience $$E$$ w.r.t. some task $$T$$ and some performance $$P$$, if its performance $$P$$, as measured by $$T$$, improves with experience $$E$$. 

  i.e., 关于任务$$T$$的程序$$Pr$$ 的表现$$P$$， 正比于从$$T$$中学习到的经验$$E$$. 

  $$P(Pr_T) \propto E(Pr_T)$$

通俗的讲，对于一个任务及其表现的度量方法，设计一种算法，让算法能够提取中数据所蕴含的规律，这就叫机器学习。如果数据带有标签，则称作监督学习；对应的，如果是无标签的，则称无监督学习。
  
<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ 符号主义人工智能
 </h3> 
  * Symbolic AI: 通过编写足够多的明确规则来处理知识，就可以实现与人类水平相当的AI. 

<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ AI V.S. Machine Learning 
 </h3> 
一图胜千言(source: Deep Learning by Iran Goodfellow et al)

<img class="center" src="{{site.baseurl}}/images/dl.vs.ai.png" width="60%">

人工智能本质上是一种智能，通常我们以智能系统或硬件的形式去描述与理解。机器学习则是实现AI的一种途径，也是当下比较主流的方式。

<!-- <h1 style="font-size:bold;">🌳 机器学习的分类 </h1>
<h2 style="font-size:bold; font-family: STHeiti, SimHei, STKaiti, KaiTi">🌲🌲 机器学习的分类 </h2>
<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ 机器学习的分类 </h3> -->


<h2 style="font-size:bold; font-family: STHeiti, SimHei, STKaiti, KaiTi">🌲🌲 机器学习的分类 
 </h2> 
依照数据类型的不同，学习算法可以大致分为以下几类

<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ 监督学习
 </h3> 
特点：学习示例中的数据都存在正确的标签

场景：分类与回归问题

常用算法：K-Nearest Neighborhood(KNN), 决策树(Decision Tree), 随机森林(Random Forest), 朴素贝叶斯(Naive Bayes), 支持向量机(Support Vector Machine, SVM), 逻辑回归(Logistic Regression), AdaBoost以及线性判别分析(Linear Discriminant Analysis, LDA)。
深度学习(Deep Learning)也是大多数以监督学习的方式呈现

<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ 无监督学习
 </h3> 
特点：数据没有标签，学习模型的目的为了推断数据的一些内在结构

场景：关联规则的学习以及聚类等

常见算法：Apriori算法以及k-Means算法

<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ 半监督学习
 </h3> 
特点：输入数据部分被标记，部分没有被标记，这种学习模型可以用来进行预测。

场景：包括分类和回归，算法包括一些对常用监督式学习算法的延伸，通过对已标记数据建模，在此基础上，对未标记数据进行预测。

​常用算法：图论推理算法（Graph Inference）或者拉普拉斯支持向量机（Laplacian SVM）

<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ 强化学习
 </h3> 
特点：智能体(agent)接收有关其环境(environment)的信息，并学会选择使某种奖励最大化的行动。 


<h2 style="font-size:bold; font-family: STHeiti, SimHei, STKaiti, KaiTi">🌲🌲 损失函数 
 </h2> 
损失函数（Loss Function，又称误差函数(error function)或者代价函数(cost function)，通过估算模型的预测值与真实值的不一致程度来衡量算法的运行效果。
损失函数是一个非负实值函数，通常使用$$\mathcal{L}(Y, f(x))$$来表示。
函数值越小，表明模型的鲁棒性就越好。损失函数是经验风险函数的核心部分，也是结构风险函数重要组成部分。


<h2 style="font-size:bold; font-family: STHeiti, SimHei, STKaiti, KaiTi">🌲🌲 梯度下降
 </h2> 

<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ 为什么需要梯度下降
 </h3> 

多数学习算法都涉及到某种形式的优化，即寻找最优参数$$x$$的值以最小化某个函数$$f(x)$$的任务(最大化问题可以通过反转$$f(x)$$的符号转换)。梯度是张量运算的导数，是导数这一概念向多元函数导数的推广。对于给定函数$$f(x)$$，梯度表示的是各点处的函数值减小最多的方向。
尽管这个方向未必指向最小值，但沿着这个方向可以最大限度减小函数的值。

而梯度下降法(Gradient Descent, GD)是常见优化算法之一。


<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ 特点
 </h3> 
简单、有效，对于凸函数(convex)来说，GD 总能很快找到最小值。但相应的，对于非凸函数(non-convex)，GD 可能会陷入到一个局部最小值，而无法收敛到全局最小值。

<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ 
 </h3> 

<h1 style="font-size:bold;">🌵数据表示与特征工程
 </h1> 

* 什么是one-hot encoding? 为什么需要它?
  * 也称N取一编码(one -out-of N encoding)，或者虚拟变量(dummy variable)。背后的思想是将**一个分类变量替换一个或者多个特征** ，新特征取值0或者1. 
  * 算法(e.g., LR)不能处理非数值型分类变量时，需要对这些变量进行编码。比较直观的做法是采用**integer encoding** (or label encoding, 整数编码)，将变量映射到一个连续数值集，e.g., [1,2  ,3,4,…]。 这种方法的弊端在于：算法可能认为数值相近的变量有相关联系，比如{'black': 1, 'red':2, 'blue':3, gray': 4}，'black'编码后的值与'red'编码后相近，但实际上'black'与'gray'更为相关。 
* 如何理解过拟合？
  * 从直观表现上来说，模型过度关注于训练集本身，在训练集上表现好，但在测试集上表现不好，泛化性能差。
  * 产生的原因：
    * 模型本身过于复杂，以至于拟合了训练样本集中的噪声。此时需要选用更简单的模型，或者对模型进行裁剪。
    * 训练样本太少或者缺乏代表性。此时需要增加样本数，或者增加样本的多样性。
    * 训练样本噪声的干扰，导致模型拟合了这些噪声，这时需要剔除噪声数据或者改用对噪声不敏感的模型。



<h1 style="font-size:bold;">🌵体系
 </h1> 

* 什么是AI ?
  * AI的本质是, 人类所赋予机器(or 工具)的一种能够解决具备一定难度的问题的能力(the ability to solve certain difficult problems — difficult in the sense that they are not easily solved by humans)，这种能力越强，就能越好的解决问题(e.g., 准确率更高的分类器)。 



<h1 style="font-size:bold;">🌵数据处理
 </h1> 

* 为什么要对数据进行归一化处理？
  * 做为一个特征，我们希望看到数据的相对值差别对结果的影响，而不是其绝对值。特别地，未做归一化的数据中，取值范围最大的数据将主导诸如kNN算法的结果
* 通俗解释过拟合、欠拟合？
  * 前者指一个模型过分关注训练数据，但对新数据的泛化性能不好，后者指模型无法获取数据中的所有变化。
* kNN算法优缺点？
  * 优势 精度高、对异常值不敏感、无数据输入假定 
  * 缺点 计算、空间复杂度高，无数给出数据的内在含义
* 决策树优缺点？
  * 计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特征数据
  * 得到的模型很容易可视化，非专家也很容易理解
  * 算法完全不受数据缩放的影响。每个特征被单独处理，数据的划分也不依赖于缩放，因此决策树算法不需要特征预处理，比如归一化或者标准化
  * 缺点 容易过拟合
* Entropy ?
  * 信息增益(information gain)，指对数据集进行处理之前之后发生的变化。对一个符号$$x$$ 的信息定义为 $$-\text{log}_2p(x)$$ 。这是一个 xxx 
  * 熵定义为信息的期望值 $$ H = - \Sigma_i^n p(x_i) \text{log}_2p(x_i)$$

* 线性回归 ?
  * 也称普通最小二乘法(ordinary least squares, OLS)，回归问题最简单也最经典的线性方法。线性回归寻找参数 $$w$$与$$b$$，使得对训练集的预测值与真实的回归目标值 $$y$$之间的**均方误差**最小。
* 岭回归(ridge regression)
  * 对于高维数据集(即有大量特征的数据集)，线性模型过拟合的可能性变大。 在岭回归中，对系数$$w$$的选择不仅要在训练数据上得到好的预测结果，还要**拟合附加约束**(E.g., 正则化)。 
  * Ridge模型在模型的简单性(系统都接近于0)与训练集性能之间做出权衡。



<h1 style="font-size:bold;">🌵决策树、随机森林
 </h1> 

对数据反复进行递归划分，直到每个区域（叶结点）只包含单目标值（单一类别或单一回归值）。 

通常来说，构造决策树直到所有叶结点都是纯的，这会导致模型非常复杂，并且对训练数据高度拟合。典型的特征是：决策边界过于关注远离同类别样本的单个异常点。 这也是决策树的一个主要缺点之一。

防止过拟合：

1. 预剪枝(pre-pruning)：限制树的最大深度、叶结点的最大数目、规定一个结点中数据点的最少数据数目
2. 后剪枝(post-pruning) 先构造树，随后删除或折叠信息量很少的结点

决策树的优点： 

1. 

为了克服决策树过拟合的缺点，一个思路是合并多个决策树，即是构建：

<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ 随机森林
 </h3> 

随机森林本质是：**多个决策树的组合**。 背后思想：每棵（决策）树的预测可能都相对较好，但可能对部分数据过拟合，如果构造很多树，并且每棵树的预测都很好，但以不同的方式过拟合，那么对这些树的预测结果取平均值来降低过拟合（对于分类问题，可以采用“软投票(soft voting)”策略，即每个算法做出“软”预测，给出每个可能输出label的概率，所有概率求平均值，输出概率最大标签）。

随机化方法

* 通过选择用于构造树的数据点，比如使用**自助采样(bootstrap sample)**, 从n_samples个数据点中有放回地随机抽取样本
* 通过选择每次划分测试的特征，每个树随机选择特征的一个子集。潜在问题 a. max_features 过大，比如等于n_features，那么所有树都考虑了全部特征，那么将十分相似 ; b. Max_features过小，比如1，为了更好拟合数据，每棵树都很深

随机森林也可以给出特征重要性(由所有树的特征重要性求和再平均)，一般来说，比单棵树给出的可为可靠。

#### 优、缺点

* 方法强大：通常不需要反复调节参数就可以得到很好的结果，也不需要对数据进行缩放。有决策树所有优点，也弥补了其过拟合的缺陷。但如果需要以可视化方式向非专家总结预测过程，选单个决策树可能更好。
* 支持多核并行， n_jobs = 9 or -1 
* 对于维度非常高的稀疏数据（比如文本数据），RF表现往往不是很好，线性模型可能更适合。

<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ 梯度提升机（梯度提升回归树）
 </h3> 

采用连续的方式构造树，每棵树都试图**纠正前一棵树的错误**。优势，通常使用深度很小(1~5)的树，占用内存少，预测速度更快。

背后思想：**合并多个简单的模型(弱学习器)，比如深度较小的树。每棵树只能对部分数据做出好的预测，通过添加更多的树，不断迭代提高性能**。 

优点：

* 深度很小、占用内在少、预测速度很快 ; 表现很好
* 不需要数据缩放

缺点：

* 需要仔细调参，训练时间可能会比较长
* 不适用于高维稀疏数据

它对参数设置比rf更为敏感，如果设置得体，精度很高。故经常是ML竞赛优胜者。

<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ 核支持向量机
 </h3> 

Kernelized support vector machine 通常简称 svm

<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ 神经网络
 </h3> 

Q:  缺点？

* 功能越强大的神经网络，通常需要更长的训练时间; 还需要仔细的预处理数据
* 调参是一门艺术
* 在“均匀”的数据上表现良好，即特征都具有相似的含义。 如果数据包含不同种类的特征，那么基于树的模型可能表现的更好。 



<h1 style="font-size:bold;">🌵无监督学习
 </h1> 

* 如何评估无监督学习?
  
  * 通常来说，评估监督算法的唯一方法就是 **人工检查 **。 
  
  



<h1 style="font-size:bold;">🌵CNN 卷积神经网络
 </h1> 

* “学习” 指以最小化损失函数为基准，从训练数据中自动获取最优权重参数的过程。自动获取的方法：梯度下降法。
* 优点：对所有的问题都可以用同样的流程来解决，可以直接将数据作为原始数据，进行“端到端”的学习。
* 损失函数
  * 可以使用任意函数，但一般用均方误差和交叉熵误差等
    * 交叉熵误差 $$E = - \sum\limits_k t_k log y_k$$ , $$t_k, y_k$$ 监督数据、神经网络的输出
  * 刻画的神经网络**性能**的“恶劣程度“
  * 梯度，（损失）函数在各个变量上的偏导数构成的向量。
* CNN(convolutional neural networks) 
  * CNN的基石是卷积层(convolutional layer)。Input layer的接收野(receptive fields, 一个$$m \times n$$ 像素矩阵)连接第一卷积层的一个神经元，同样，第1卷积层的一个 $$k \times l $$ 神经元矩阵连接到第2层卷积层。因此，第$$i$$层卷积层的第 $$j, j+1$$个神经元所对应的第 $$i-1$$ 层神经元矩阵可能是重叠的($$k = l = 1$$就不重叠)。有时候为了使第 $$i, i+1$$ 卷积层长宽一致，会在卷积层周围添加一些 0, 称为**0填充**. 
* 下采样？
  * 减少需要处理的特征图的元素个数 ； 通过让连续卷积层的观察窗口越来越大，从而引入空间过滤器的层级结构。 
  * 最大池化、平均池化、步幅都可以实现下采样； 最大池化效果通常较好。 
* VGG-16 v.s. VGG-19 
  * VGG-16 结构图 <img src="{{site.baseurl}}/images/2019/vgg16.png" width="200px">
  * VGG-16 包含16层深度神经网络层，VGG-19 19 层 
* 全连接层存在的问题
  * 数据的形状被”忽视“。图像有3维特征，长、宽、通道，但向全连接层输入时，需要将3维数据拉开为1维，这将忽略重要的空间信息，比如：空间上邻近像素有相似的值、RBG各通道之间分别有密切的关联性、相距较远像素之间几乎没有关联 etc



<h1 style="font-size:bold;">🌵深度学习
 </h1> 

<h2 style="font-size:bold; font-family: STHeiti, SimHei, STKaiti, KaiTi">🌲🌲 文本与序列
 </h2> 

* 文本向量化(vectorize)实现方法 
  * 将文本转化成数值张量的过程，有以下几种实现方式： 
  * 将文本分割成**单词**，并将每个单词转换成一个向量
  * 将文本分割成**字符**，并将每个字符转换成一个向量
  * 提取单词或字符的**n-gram**（多个连续单词或者字符的集合）, 并将每个n-gram转换为一个向量。

* 标记(token) 与分词(tokenization) ?
  * Token，将文本分解而成的单元（单词、字符或n-gram)
  * Tokenization，文本分解成标记的过程。 所有vectorize过程都是应用某种分词方案，然后将数据向量与生成的标记相关联。
* 关联向量与标记的方法 ？
  * **One-hot编码**：每个单词(或字符)与一个唯一的整数索引相关联，然后将这个整数索引 $i$  转换成长度为 $N$（词表大小） 的二进制向量。
  * **词嵌入(word embedding)**: xxx 
    * 完成主任务的同时学习词嵌入
    * 预训练词嵌入(pretrained word embedding)，在不同于待解决问题的机器学习任务上预计算好词嵌入，然后将其加载到模型中
  * 区别：
    * one-hot 二进制、稀疏、高维度(维度大小=词表中单词个数)、硬编码； 
    * 词嵌入 密集、低维、从数据中学习得到 

* 