---
layout: post
title: Understanding Word2Vec
date: 2020-04-24
tags: python word2vec gensim
categories: machine_learning
author: GaoangLau
---
* content
{:toc}


# Word Embedding and Word2Vec 




Word2Vec, developed by [Tomas Mikolov in 2013 at Google](https://arxiv.org/pdf/1310.4546.pdf), is one of the most popular techniques to learn word embeddings using shallow neural network. Word2Vec can be obtained using two methods (both involving Neural Networks): using context to predict a target word (a method known as **continuous bag of words, or CBOW**), or using a word to predict a target context, which is called **skip-gram**.



## Word2Vec 
Word2Vec is a group of related models that are used to produce word embeddings. These models are shalow, two-layer neural network whose input is a **text corpus** and output is a vector space: **feature vectors that represent words in that corpus**. The output vectors can be fed to deep neural networks for further training.

The purpose and usefulness of Word2vec is to **group the vectors of similar words together in vector space**.
The idea behind is fairly straightforward: **you shall know a word by the company it keeps**([Firth, J. R. 1957:11](https://en.wikipedia.org/wiki/John_Rupert_Firth)). This is analogous to the saying show me your friends, and I'll tell who you are [1].
Thus words having similar neighbors, i.e., the usage context is about the same, are highly possible having same meaning or at least highly related. For example, 'possible' and 'probably' are typically used in the same context. 

Word2vec works like an auto-encoder, encoding each word in a vector, but rather than training against the input words through reconstruction, as a [restricted Boltzmann machine](https://pathmind.com/wiki/restricted-boltzmann-machine) does, word2vec trains words against other words that neighbor them in the input corpus.

## Natural Wold Embeddings
<!-- Loosely speaking, **word embedding** are vector representations of a particular word.  -->
The vectors we use to represent words are called **neural word embeddings**. Thus, with the "vectorization" procedure, words in text corpus are now (vectors of) numbers in word embeddings.


# Pre-trained models
Gensim (found below section for a Quickstart tutorial) launched its own dataset storage on [Github](https://github.com/RaRe-Technologies/gensim-data), committed to long-term support, a sane standardized usage API and focused on datasets for unstructured text processing. 

This repo contains the following pre-trained models:
* fasttext-wiki-news-subwords-300 
* glove-twitter-100 
* glove-twitter-200 
* glove-twitter-25 
* glove-twitter-50 
* glove-wiki-gigaword-100 
* glove-wiki-gigaword-200 
* glove-wiki-gigaword-300 
* glove-wiki-gigaword-50 
* word2vec-google-news-300 
* word2vec-ruscorpora-300 

Load a model is simple with `Gensim` downloader API:
```python
import gensim.downloader as api
model = api.load('word2vec-google-news-300)
model.most_similar('cat')
```
 



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
5. [A Beginner's Guide to Word2Vec and Neural Word Embeddings](https://pathmind.com/wiki/word2vec)