---
layout:     post
title:      Skip-gram with negative sampling
date:       2023-09-30
tags:   [skip-gram, embedding]
categories: 
- nlp
---

关于 [skip-gram](https://arxiv.org/abs/1301.3781) 的原理介绍，已经有很多文章了，这里不再赘述。本文主要介绍 skip-gram with negative sampling (SGNS) 的原理。


首先，简要回顾一个 skip-gram 的模型结构，如下图所示：
<div style="display: flex; justify-content: center;">
  <div style="margin-right: 10px;">
    <img src='https://file.ddot.cc/imagehost/2023/cbow-skipgram.png' width='500pt'>
  </div>
</div>

Skip-gram 的工作机制，简单来说是，是给定中心词，然后去预测它的上下文。比如：

```text
The quick brown fox jumps over the lazy dog
```

给定 `fox`，然后去预测它的上下文 `The quick brown jumps over the lazy dog`。在训练时，我们会给定一个窗口大小 $c$，然后从中心词的左右两边各取 $c$ 个词，这些词就是中心词的上下文。比如，给定中心词 `fox`，窗口大小 $c=2$，那么它的上下文就是 `The quick brown jumps`。 两两组合，我们就可以得到正样本，比如：(fox, quick), (fox, brown), (fox, jumps), (fox, over)。

余下的词，用于负样本，比如：(fox, the), (fox, lazy), (fox, dog)。
可以看出，当文本长度很长时，负样本的数量会远远大于正样本的数量，这样会导致模型训练的很慢，所以，我们需要一种方法来减少负样本的数量，这就是 negative sampling 的作用。

# 如何选负样本 

采样是根据 unigram distribution 来进行的，也就是根据单词的概率，出现概率高的单词容易被选为负样本。

$p(w_i) = \frac{f(w_i)}{\sum_j f(w_j)}$

在论文([《Distributed Representations of Words and Phrases
and their Compositionality》](https://arxiv.org/pdf/1310.4546.pdf))中，作者声称使用下面的公式结果更好，会增加一些出现频率少的单词被选中的概率，减小常见单词被选中的概率。

$p(w_i) = \frac{f(w_i)^{3/4}}{\sum_j f(w_j)^{3/4}}$

[END]