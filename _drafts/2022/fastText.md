---
layout:     post
title:      fastText 
date:       2022-09-03
tags: [nlp,fastText]
categories: 
- nlp 
---

[fastText](https://fasttext.cc/docs/en/support.html) 文本分类算法是由 Facebook AI Research (FAIR) 开发的一个高效的用于**文本分类**及**词表示**的库。


## 文本分类 

在 NLP 中，文本分类有多种算法，TextCNN，RNN，LSTM，基于 Transformer 的预训练模型，还有最近比较热门的基于提示 (prompt learning) 的算法。 

文本分类是 fastText 提供的两大功能之一，对比其他分类算法， fastText 的优势在于：速度快且效果佳，是难得的在速度上及效果上都能达到高水准的一种分类算法。它的思想与比较简单，fastText 由三层网络组成，分别是输入层，隐层，输出层。输入层是一个词向量，隐层是一个线性层，将输入层的词向量做一个平均，输出层是一个 softmax 层。

<img src="https://raw.githubusercontent.com/117v2/stuff/master/2022/fastText1.png" width=50%>

整个模型结构与 word2vec 中的 CBOW 模型类似，区别在于 ：
1. fastText 使用上下文预测文本的类别。 
2. CBOW 以词为单位， 而 fastText 以若干字符为单位。 这样的一个好处是，fastText 可以很好的处理未登录词 (out-of-vocabulary, OOV)，因为它能从 OOV 的 n-gram 中得到词的一个表示，比如 pricess 是一个 OOV，但语料中出现过 price 和 waitress，那么 fastText 就能从 price 和 ess 中得到一个 pricess 表示的一个近似。

### 预训练词向量  
在进行分类过程中，也可以使用预训练词向量。当训练数据集比较小时，使用预训练词向量开始训练模型，可以提供一些先验知识。但需要注意的是，预训练词向量的语料与训练数据集的语料要尽量保持一致，否则会导致模型的效果下降。

使用方法如下： 
```python
model = fasttext.train_supervised(input=TRAIN_FILEPATH, lr=1.0, epoch=100,
                             wordNgrams=2, bucket=200000, dim=300, loss='hs',
                             pretrainedVectors=VECTORS_FILEPATH)
```                             
fastText 提供了 157 种语言的预训练词向量，详情可参考： https://fasttext.cc/docs/en/crawl-vectors.html


## 学习词表示 
与 word2vec 类似，fastText 也提供了 CBOW 和 Skip-gram 两种学习词表示的方法。

与 word2vec 不同的是，在word2vec中，我们并没有直接利⽤构词学中的信息。⽆论是在跳字模型还是连续词袋模型中，我们都将形态不同的单词⽤不同的向量来表⽰。比如 "cat", "cats" 分别⽤两个不同的向量表⽰，而模型中并未直接表达这两个向量之间的关系，尽管最后得到的词向量比较接近。 鉴于此，fastText提出了⼦词嵌⼊(subword embedding)的⽅法，从而试图将构词信息引⼊word2vec中的CBOW。

## 动手实现一个简单版本 
fastText 的源码是用 C++ 实现的，它也提供了 Python 接口，可以直接使用。
如上所述，fastText 的核心是一个三层的神经网络，我们使用 Keras 实现一个简单版本的 fastText，使用 `keras.layers.Embedding` 层将输入的词转换为词向量，然后使用 `keras.layers.GlobalAveragePooling1D` 层将词向量求平均，最后使用 `keras.layers.Dense` 层进行分类。 代码如下： 

```python
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (Dense, Embedding, GlobalAveragePooling1D,
                                     TextVectorization)
from tensorflow.keras.models import Sequential

def myfastText(df: pd.DataFrame,
               embedding_dims: int = 100,
               ngrams: int = 2,
               max_features: int = 30000,
               maxlen: int = 400,
               batch_size: int = 32,
               epochs: int = 10):
    """ A simple implementation of fastText.
    """
    X, Xv, y, yv = train_test_split(df['text'], df['target'], random_state=0)
    args = {
        'dim': embedding_dims,
        'ngrams': ngrams,
        'max_features': max_features,
        'maxlen': maxlen,
        'batch_size': batch_size,
        'epochs': epochs
    }
    logger.info(args)
    vectorize_layer = TextVectorization(
        ngrams=ngrams,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=maxlen,
    )
    vectorize_layer.adapt(X.values)
    model = Sequential([
        vectorize_layer,
        Embedding(max_features + 1, embedding_dims), # 词向量层
        GlobalAveragePooling1D(), # 池化层
        Dense(3, activation='softmax') # softmax 输出层
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    history = model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(Xv, yv),
    )
    return model, history
```

## 特点 
1. 分类速度快。根据[官方给出的数据]( https://fasttext.cc/blog/2016/08/18/blog-post.html)，使用标准多核 CPU fastText 在 10 分钟内可以在超过 10 亿词的语料上完成训练，5 分钟内在 50 万条语句上完成 30 万个类别的分类任务。 
2. 支持多种语言：利用其语言形态结构，fastText能够被设计用来支持包括英语、德语、西班牙语、法语以及捷克语等多种语言。FastText的性能要比时下流行的word2vec工具明显好上不少，也比其他目前最先进的词态词汇表征要好。
专注于文本分类，在许多标准问题上实现当下最好的表现（例如文本倾向性分析或标签预测）



## 总结
fastText 本质上是一个浅层的神经网络模型，使用 n-gram 特征作为输入，通过更细粒度的 token 学习词表示。 fastText 运行速度快，且在小数据集上效果也不错，通常情况下，可以作为一个不错的 baseline 模型使用。

## 参考资料
- [fastText support](https://fasttext.cc/docs/en/support.html)
- [fastText: Library for efficient learning of word representations and sentence classification](https://arxiv.org/abs/1607.01759)
