---
layout: post
title: Semantic Similarity
date: 2022-10-18
tags: nlp similarity
categories: nlp
author: berrysleaf
---
* content
{:toc}


语义相似度有很多重要的应用场景，比如在检索系统中用来做语义召回，或者作为精排的特征。基于文本语义相似度模型做相似检索可以辅助文本分类，能够弥补分类模型更新迭代周期长的问题。在智能问答系统中，文本语义相似度模型也能发挥很大的作用。




# 文本相似度模型的发展历程
从最早的基于词频的相似度计算方法，到基于统计语言模型的相似度计算方法，再到基于深度学习的相似度计算方法，文本相似度模型的发展历程如下图所示。

<img src="https://image.ddot.cc/202311/semantic_similarity_map_rc.png" width=789pt>


学习相似度的深度学习范式主要有两种，如下图所示

第一种范式是先通过**深度神经网络模型提取输入的表示向量**，再通过表示向量的简单距离函数（eg. 内积，欧式距离等）计算两者的相似度。这种方式在提取表示向量的过程中只考虑当前输入，不考虑要与之计算相似度的另一个输入的信息，通常用孪生网络来实现。属于这一类的常用模型包括 DSSM、ARC-I、CNTN 等。

第二种范式是通过深度模型提取**两个输入的交叉特征，得到匹配信号张量，再聚合为匹配分数**，该方式同时考虑两个输入的信息，因而一般情况下效果相比第一种范式要更好，不足之处在于预测阶段需要**两两计算相似度，计算空间很高**，因而不适合用来做大规模召回，只能用在精排阶段。ARC-II、MatchPyramid、Match-SRNN、Duet 等模型都属于这一类型。


# 评价指标 
首先，对于每一个文本对，采用余弦相似度对其打分。打分完成后，采用所有余弦相似度分数和所有 gold label 计算 Spearman Correlation。

其中，Pearson Correlation 与 Spearman Correlation 都是用来计算两个分布之间相关程度的指标。Pearson Correlation 计算的是**两个变量是否线性相关**，而 Spearman Correlation 关注的是**两个序列的单调性是否一致**。Pearson Correlation 与 Spearman Correlation 的公式如下:

$$\rho_{XY} = \frac{cov(X,Y)}{\sigma_X \sigma_Y} \ \textrm{(Pearson Corr)}$$

$$\rho_{s} = 1 - \frac{6 \Sigma_i^n d_i^2}{n(n^2-1)} \ \textrm{(Spearman Corr)}$$


# 经典模型
## Jackard 相似度
Jackard 相似度是最简单的文本相似度计算方法，思路是统计两段文本共同的独特词汇的数量。

$$J(A,B) = \frac{\vert A \cap B \vert}{ \vert A \cup B \vert}$$

其中，$$A$$ 和 $$B$$ 分别是两个集合，$$\vert A \cap B \vert$$ 表示两段文本独特词的交集，$$\vert A \cup B\vert$$ 表示两段文本独特词的并集。$$A \cup B$$ 用作归一化，避免文本过长导致的相似度偏高。

一般情况下，计算 Jackard 相似度时使用的是 1-gram，即将文本分词后，统计词频。也可以使用 2-gram，即统计相邻两个词的频率，或者更高的 n-gram。采用 n-gram 的方法也称为 [w-shingling](https://en.wikipedia.org/wiki/W-shingling)。

```python
# A Python implementation of Jackard similarity
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))
list1 = ['dog', 'cat', 'cat', 'rat']
list2 = ['dog', 'cat', 'mouse']
jaccard_similarity(list1, list2) # 0.5
```

## 词袋模型
词袋模型（Bag of Words, BoW）是经典的文本表示方法，通过统计词频等方式从文本中提取特征，将文本转换为向量。通过计算向量间的距离，可以得到文本间的相似度。比较流行的方式有 CountVectorizer 和 TfidfVectorizer。

```python
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
# Load the English STSB dataset
stsb_dataset = load_dataset('stsb_multi_mt', 'en')
stsb_train = pd.DataFrame(stsb_dataset['train'])
stsb_test = pd.DataFrame(stsb_dataset['test'])

model = TfidfVectorizer(lowercase=True, stop_words='english')

# Train the model
X_train = pd.concat([stsb_train['sentence1'], stsb_train['sentence2']]).unique()
model.fit(X_train)

# Generate Embeddings on Test
sentence1_emb = model.transform(stsb_test['sentence1'])
sentence2_emb = model.transform(stsb_test['sentence2'])

# Cosine Similarity
stsb_test['TFIDF_cosine_score'] = cos_sim(sentence1_emb, sentence2_emb)
```

词袋方法存在的问题是：
- 如果文档规模很大，词汇量也会很大，那么这种方法产生的向量维度很高。
- 通常情况下，一个文档中仅出现一小部分词汇，向量高度稀疏。


## WMD
Jaccard 方法及词袋模型基于的假设是**相似的文本包含很多相同的词汇**。这种假设在实际应用中不成立，比如，以下两段文本中的词汇完全不同，但是两段文本的主题是相同的。

```python
text1 = "Obama speaks to the media in Illinois"
text2 = "The President greets the press in Chicago"
```
**相似的文本应该有相同的语义**，而词向量可以作为词的分布式语义表示。如果从词的语义上着手计算，可以得到更好的文本相似度计算方法。

[Word Mover's Distance (WMD)](http://proceedings.mlr.press/v37/kusnerb15.pdf) 基于 Earth Movers Distance 的概念，刻画的是一个文档的词嵌入（embeddings） "旅行" 到另一个文档的词嵌入的最小距离。由于每个文档包括多个词，WMD 计算需要计算每个词到其他每个词的距离。它还会根据每个词频对 "旅行" 距离进行加权。gensim 库提供了一个快速计算 WMD 的接口。 

WMD 效果要优于 Jaccard 相似度和词袋模型，但同样没有考虑到上下文语境，对于不同的语境，相同的词语可能有不同的语义。

```python
from gensim.models import KeyedVectors
from gensim.similarities import WmdSimilarity
# Load Google's pre-trained Word2Vec model.
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

text1 = 'Obama speaks to the media in Illinois'
text2 = 'The president greets the press in Chicago'
instance = WmdSimilarity([text1, text2], model, num_best=2)
# Query
query = "Obama speaks to the media in Illinois"
sims = instance[query]  # A query is simply a "look-up" in the similarity class.
print(sims)  # prints [[(0, 0.99999994), (1, 0.0)]]
```

- Python 实现: https://github.com/hechmik/word_mover_distance

### 质疑
也有人质疑 WMD 原始论文中的结论，认为:
1. 在使用 L1-正则情况下， WMD 与 L1-normalized BOW 效果相当
2. WMD 有效的原因不是因为 *taking the underlying geometry into account*，而是因为对向量进行了正则。

质疑文章链接: https://proceedings.mlr.press/v162/sato22b/sato22b.pdf



# 基于上下文的文本相似度计算
## USE 
通用句子编码器（[Universal Sentence Encoder](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder), USE ）能够直接获取句子的嵌入编码，可以简单地用于计算句子层面的语义相似性，也可以结合少量的监督训练数据来提升下游分类任务的效果。

> USE paper: [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)

## Sentence Transformers <span id="sbert"></span>
[Sentence Transformers](https://arxiv.org/pdf/1908.10084.pdf)，也称为 Sentence BERT（SBERT），因为这一系列模型都是基于 BERT 的。训练时通过对比学习的方式，将句子映射到同一空间，使得相似的句子在空间中距离更近，不相似的句子距离更远。

### Bi-Encoders and Cross-Encoders
在 https://www.sbert.net/examples/applications/cross-encoder/README.html 中提到 SBERT 有两种模型结构，分别是 Bi-Encoders 和 Cross-Encoders。

Bi-Encoders(Sentence Bert, SBert) 对输入的句子对，分别通过 BERT 编码器编码后，再通过 mean pooling 池化获得句子的向量表示，然后通过计算余弦相似度得到句子的相似度。

<img src="https://image.ddot.cc/202311/bi-encoders_rc.png" width=378pt>

Cross-Encoders 同时将两个句子都输入到 BERT，编码之后经过一个分类头输出 0-1 之间的相似度。不产生中间的句子向量表示。

<img src="https://image.ddot.cc/202311/Bi_vs_Cross-Encoder_rc.png">

Cross-Encoders 相较于 Bi-Encoders 性能较好，但是计算速度慢，因为需要同时编码一个句子对，而 Bi-Encoders 只需要逐个编码每一个句子。举例来说，如果当下有 10000 条语句，Bi-Encoders 需要编码 10000 次，而 Cross-Encoders 需要编码 5 千万个句子对。

<img src="https://image.ddot.cc/202311/cross-encoders_rc.png" width=378pt>

Sentence-Transformers 的[文档](https://www.sbert.net/examples/applications/cross-encoder/README.html)中也提到了一个结合使用 Bi-Encoders 和 Cross-Encoders 的方法，对某些场景下，比如信息检索或者语义搜索，可以使用 Bi-Encoders 快速召回一批候选句子，然后再使用 Cross-Encoders 重排句子的相似度。 

## SimCSE <span id="simcse"></span>
[SimCSE](https://arxiv.org/pdf/2104.08821.pdf) 是指简单的句子嵌入对比学习（Simple Contrastive Learning of Sentence Embeddings），对比学习的思想是拉近相似样本，推开不相似的样本。SimCSE 有两种训练方法，一种是无监督训练方法，一种是有监督训练方法。无监督部分核心的创新点在于采用了 Dropout 的方法添加噪音进行了文本增强。

SimCSE 的训练思路是：
- 给定一个文本文件，使用任何预先训练好的 BERT 模型作为编码器计算该文本的嵌入，并取 [CLS] 标记的向量。
- 对以上向量使用两个不同的 Dropout 掩码，创建两个噪声文本嵌入。这两个从同一输入文本中产生的噪声嵌入被认为是一对 "正例"，模型希望它们的余弦距离为 0。 
- 将该批文本中所有其他文本的嵌入视为"负例"。模型希望 "负面的" 与上一步的目标文本嵌入的余弦距离为 1。损失函数然后更新编码器模型的参数，使嵌入更接近我们的预期。
- 监督学习场景中 SimCSE 结合使用自然语言推理（NLI）标记的数据集，将其中被标记为 "entailment" 的文本作为正例对，从被标记为 "contradiction" 的文本对作为负例。 

SimCSE 整体结构如下图所示：

<img src="https://image.ddot.cc/202311/simcse_process_rc.png" width=678pt>

Supervised SimCSE 的损失函数：

$$\begin{aligned}
-\log \frac{e^{\text{sim}(h_i, h_i^+)/\tau}}{\sum_j^N(e^{\text{sim}(h_i, h_j^+)/\tau} + e^{\text{sim}(h_i, h_j^-)/\tau})}
\end{aligned}$$

其中，$$h_i$$ 表示原始文本的嵌入，$$h_i^+$$ 对应句与 $$x_i$$ 为蕴含关系，$$h_i^-$$ 对应句与 $$x_i$$ 为矛盾关系，$$N$$ 表示批次中的文本数量，$$\tau$$ 表示温度参数，$$\tau$$ 参数的值越小，相似度分布的“温度”越低，表示对相似度的判别更加严格。意味着在训练时，相似的句子会更有可能被赋予较高的概率，而相似度较低的句子则会有较低的概率。通过调节 $$\tau$$ 参数，可以影响相似度分布的平滑程度，从而对模型的训练产生影响。



### 横向对比 
- SimCSE 与 SimBERT 的区别

苏老师在[博文](https://spaces.ac.cn/archives/8348)中提到，SimCSE 可以看成是 SimBERT 的简化版（[《鱼与熊掌兼得：融合检索和生成的SimBERT模型》](https://spaces.ac.cn/archives/7427)），它简化的部分为：
1. SimCSE 去掉了 SimBERT 的生成部分，仅保留检索模型
2. 由于 SimCSE 没有标签数据，所以把每个句子自身视为相似句传入。


## CoSENT

<span style="color:red">Co</span>sine <span style="color:red">Sent</span>ence，由苏老在[CoSENT（一）：比Sentence-BERT更有效的句向量方案](https://kexue.fm/archives/8847)中提取，实验表明，CoSENT 在收敛速度和最终效果上普遍都比 InferSent 和 Sentence-BERT 要好。

<img src="https://image.ddot.cc/202311/cosent_result_rc.png" width=789pt>



## 小结 
基于 BERT 的方法外推性不好，不支持长度大于 512 的句子。对于英文的相似度查询任务，一个方案是使用 [OpenAI 的嵌入生成接口](https://beta.openai.com/docs/guides/embeddings/what-are-embeddings) 处理最长不超过 2048 个 tokens 的句子。 


有前人总结了一份如何选择模型的 flowchart，如下图所示:

<img src="https://image.ddot.cc/202311/semantic_similarity_choice_rc.png" width=678pt>

图来源：https://towardsdatascience.com/semantic-textual-similarity-83b3ca4a840e （仅供参考，具体问题需要具体分析）


# 参考 
- [一文详解文本语义相似度的研究脉络和最新进展, CSDN](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/126357166)
- [深度学习语义相似度系列：Ranking Similarity](https://yangxudong.github.io/dsmm/)
- [Semantic Textual Similarity](https://towardsdatascience.com/semantic-textual-similarity-83b3ca4a840e)
- [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/pdf/2104.08821.pdf)
  