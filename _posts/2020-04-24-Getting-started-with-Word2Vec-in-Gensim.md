---
layout:     post
title:      Getting started with Word2Vec in Gensim 
date:       2020-04-24
tags: [Python, word2vec, gensim]
categories: 
- machine learning
---

# Word embedding and Word2Vec 
Loosely speaking, **word embedding** are vector representations of a particular word. And Word2Vec, developed by [Tomas Mikolov in 2013 at Google](https://arxiv.org/pdf/1310.4546.pdf), is one of the (most popular) technique to learn word embeddings using shallow neural network. Word2Vec can be obtained using two methods (both involving Neural Networks): **Skip Gram** and **Common Bag Of Words (CBOW)**.

The purpose and usefulness of Word2vec is to group the vectors of similar words together in vector space.
The idea behind is pretty simple. We are making an assumption that **you can tell the meaning of a word by the company it keeps**. This is analogous to the saying show me your friends, and I'll tell who you are [1].
Thus words having similar neighbors, i.e., the usage context is about the same, are highly possible having same meaning or at least highly related. For example, 'possible' and 'probably' are typically used in the same context. 

# A few pre-trained useful 
https://pathmind.com/wiki/word2vec

# TODO train your own word2vec ??



# Get started with genism
`Gensim` [2] = 'Generate Similar', is a Python library for data science. 

To remove stop words from a sentence (line):
```python
import gensim
line = gensim.parsing.preprocessing.remove_stopwords(line)
```

The stop words are saved in a `frozenset`
```python
gensim.parsing.preprocessing.STOPWORDS
```

To tokenize a sentence:
```python
gensim.utils.tokenize(line, lowercase=False) # or
gensim.utils.simple_preprocess(line)
```
Note that the later will lowercase all words. Use the first one if your task at hand is case-sensitive. 

## Train the Word2Vec model
Training a word2vec model is fairly straightforward. The following code passes `documents`, a list of tokenized document, to the method `Word2Vece`, which will create a vocabulary.

Behind the scenes we are actually training a simple neural network with a single hidden layer, of which  the weights, viewed through `model.wv`, is actually word vector we want to learn.

Some parameters:
1. `size`, dimensionality of the word vectors
2. `window` (int, optional), maximum distance between the current and predicted word within a sentence
3. `min_count` (int, optional), ignores all words with total frequency lower than this
4. `workers` (int, optional), number of threads to train the model

A more detailed document can be found [here](https://radimrehurek.com/gensim/models/word2vec.html).


```python
model.wv.most_similar(positives=['Apple'], topn=10) # look up top 10 words similar to 'Apple'
model.wv.similarity(w1="dirty", w2="smelly") # similarity between two different words
model.wv.vocab # the vocabulary, set of uniq words, been created by the Word2Vec method
```




```python
documents = ... # some tokenized film/food reviews 
model = gensim.models.Word2Vec(documents, size=150, window=10, min_count=2, workers=10)
model.train(documents,  total_examples=len(documents),  epochs=10)
```


# References 
1. [NLP in practice, Github page tutorial, Kavita Ganesan.](https://github.com/kavgan/nlp-in-practice/blob/master/word2vec/Word2Vec.ipynb)
2. [Gensim, a free python library for data science, NLP and deep learning](https://radimrehurek.com/gensim/index.html)
3. [Distributed Representations of Words and Phrases and their Compositionality, Tomas Mikolov et al.](https://arxiv.org/pdf/1310.4546.pdf)
4. [word2vec Parameter Learning Explained](https://arxiv.org/pdf/1411.2738.pdf)