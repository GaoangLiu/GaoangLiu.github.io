---
layout:     post
title:      Text processing with Python
date:       2019-09-09
tags: [tokenizer, preprocessing]
catogeries: 
- machine learning
---

# Process text with NLTK
## Tokenization
This is a process that divides big quantity of text into smaller parts called **tokens** (usually words).

Example 1: use method `word_tokenize()` to split a sentence into words:
```python
import nltk
s = "At eight o'clock on Thursday morning ... Arthur didn't feel very good."
tokens = nltk.word_tokenize(s)
# ['At', 'eight', "o'clock", 'on', 'Thursday', 'morning', 'Arthur', 'did', "n't", 'feel', 'very', 'good', '.']
```


A `RegexpTokenizer` splits a string into substrings using a regular expression. For example, the following tokenizer forms tokens out of alphabetic sequences, money expressions, and any other non-whitespace sequences:

```python
from nltk.tokenize import RegexpTokenizer
strstr = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
tkn = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
tkn.tokenize(strstr)
# ['Good', 'muffins', 'cost', '$3.88', 'in', 'New', 'York', '.', 'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']
```

Another example of a tokenizer selecting just the capitalized words:
```python
tkn = RegexpTokenizer('[A-Z]\w+')
tkn.tokenize(strstr)
# ['Good', 'New', 'York', 'Please', 'Thanks']
```


## 处理文本数据

向量与标记相关联的方式有很多种，比较常用的有
* one-hot编码(one-hot encoding)
* 标记嵌入(token embedding)，通常只用于单词，也称词嵌入(word embedding)

### one-hot encoding
以下假设每个单元都为一个单词。 

one-hot encoding 将每个标记与一个唯一的整数索引相关联，然后将这个整数索引$$i$$转换成长度为$$N$$的binary向量，这个向量只有第$$i$$个元素是1，其余为0. 

`Keras`的内置函数可以完成对原始文本数据进行单词级或字符级的one-hot编码。这些函数还提供了许多特性，比如去除特殊字符、只考虑数据集中前$$N$$个最常见单词。 

```python
samples = ['The cat sat on the mat.', 'The dog ate my homework']

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

# [[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]] 
sequences =  tokenizer.texts_to_sequences(samples)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# {'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat': 5, 'dog': 6, 'ate': 7, 'my': 8, 'homework': 9}
word_index = tokenizer.word_index
```

### 词嵌入

使用密集向量(低维的浮点数向量)来表示。

获取词嵌入的两种方法：

* 在完成主任务的同时学习词嵌入
* 预训练词嵌入：在其他任务上预计算好词嵌入，然后将其加载到模型中



对第1种方法，`keras` 提供了`Embedding`层来学习词嵌入。`Embedding`层的作用可以理解为一种字典查找，输入是一个二维张量`(samples, seq_length)`，输出三维浮点数张量 `(samples, seq_length, embedding_dimensionality)`.

> 单词索引  -> `Embedding`层 -> 对应的词向量

### 词袋

文本的一种表示方式，本质上是一个集合，其舍弃了文本中的语法结构。





### IMDB 数据集

1. [Large Movie Review Database](http://ai.stanford.edu/~amaas/data/sentiment/)
2. [Movie Review Database-2](http://mng.bz/0tIo)

Tips: 解压数据集后得到结果：

```bash
data/aclImdb/
├── test
│   ├── neg
│   ├── pos
└── train
    ├── neg
    ├── pos
```



一次性读入数据可以采用`sklearn`内置函数处理，而不用手动写函数 

```python
from sklearn.datasets import load_files
train, test = load_files("data/aclImdb/train/"), load_files("data/aclImdb/test/")
```



## 小结

