---
layout: post
title: CountVectorizer of text
date: 2020-04-23
tags: python sklearn countvectorizer
categories: machine_learning
author: GaoangLau
---
* content
{:toc}


# Basic usage 

```python
from sklearn.feature_extraction.text import CountVectorizer



doc = ["I will be having a White House Press Conference today at 5:45 P.M. Thank you!", "I have instructed the United States Navy to shoot down and destroy any and all Iranian gunboats if they harass our ships at sea."]

cv = CountVectorizer(doc)
cv.fit_transform(doc)
```
What happens when `CountVectorizer(doc)` is executed ? 

The text documents `doc` is converted into a matrix of token counts. More specifically, the method will
1. lowercases text 
2. uses `utf-8` encoding
3. performs tokenization (converts raw text to smaller units of text)
4. uses word level tokenization (meaning each word is treated as a separate token)
5. ignores single characters during tokenization (say goodbye to words like 'a' and 'I')

This implementation produces a sparse representation of the counts using `scipy.sparse.csr_matrix`.

Note that, there are some default parameters here: 
1. `encoding='utf-8'`, 
2. `lowercase=True`, convert all characters to lowercase before tokenizing. If your tasks are case-sensitive, e.g., some spam prefer uppercase titles or capitalizing words in text, you should consider setting the param to `False`. 
3. `stop_words=None`, to eliminate stop words from the text. Using stop words in `CountVectorizer` is generally a good idea. Popular stop word modules are `wordcloud.STOPWORDS` and `nltk.corpus.stopwords`. You can also customized your own stop words list like [this one](https://raw.githubusercontent.com/117ami/117ami.github.io/master/materials/stopwords.txt).
4. `strip_accents{'ascii', 'unicode', None}`. Remove accents and perform other character normalization during the preprocessing step. 'ascii' is a fast method that only works on characters that have an direct ASCII mapping. 'unicode' is a slightly slower method that works on any characters. None (default) does nothing.
5. `binary=False`, use the counts of terms/tokens. In tasks where the frequency of occurrence is insignificant, we can set `binary=True`, and thus the presence or absence of a term instead of the raw counts matters. 


# TfidfVectorizer vs. CountVectorizer
First, we should understand [`TfidfTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer).

This method transforms a count matrix to a normalized tf or tf-idf representation. Where Tf means **term-frequency** while tf-idf means **term-frequency times inverse document-frequency**. This is a common term weighting scheme in information retrieval, that has also found good use in document classification.

The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to **scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus**.

## Formula 
1. `smooth_idf=False` $$\text{tfidf}(w, d) = \text{tf(w, d)} (\text{log}(\frac{N_d}{N_w}) + 1)$$, where $$N_d$$ is the total number of documents in the document set and $$N_w$$ is the document frequency of `w`, which is the number of documents in the document set that contain the term `w`. And `tf(w, d)` is the number of `w` occurs in `d`. 

The constant `1` in `idf(w, d)` makes assure terms with zero idf, i.e., terms that occur in all documents in a training set, will not be entirely ignored. 

2. `smooth_idf=True` (default), $$\text{tfidf}(w, d) = \text{tf(w, d)} (\text{log}(\frac{1 + N_d}{1 + N_w}) + 1)$$, adding 1 to the numerator and denominator of the idf to prevent zero divisions. This can be explained as **there is an extra document that contains every term in the collection exactly once**. 

## Example 

```python
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
corpus = ['this is the first document', 
    'this document is the second document',
    'and this is the third one',
    'is this the first document']
pipe = Pipeline([('count', CountVectorizer()),
                 ('tfid', TfidfTransformer())]).fit(corpus)    
print(pipe['count'].transform(corpus).toarray())
```

`TfidfVectorizer()`, which is equivalent to `CountVectorizer` followed by `TfidfTransformer`, actually does the same thing as `pipe` does in the above code.

# References 
* [6.2. Feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
* [10+ Examples for Using CountVectorizer](https://kavita-ganesan.com/how-to-use-countvectorizer/)
* [CountVectorizer - sklearn document](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
* [TfidfTransformer - sklearn document](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer)