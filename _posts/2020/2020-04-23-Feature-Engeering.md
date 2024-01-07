---
layout: post
title: Feature Engeering
date: 2020-04-23
tags: python sklearn countvectorizer
categories: machine_learning
author: gaoangliu
---
* content
{:toc}


# CountVectorizer

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

# 结合 pipeline 使用  
```
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifiers', Xgboost())
])
pipeline.fit(X_train, y_train)
```

在一些情况下，一条输入样本可能包含多条文本 text1, text2, textn，且附带其他已有特征。如果需要对不同的 texti 使用不同的处理方式，可以参考 [stackoverflow: Using TF-IDF with other features in scikit-learn](https://datascience.stackexchange.com/questions/22813/using-tf-idf-with-other-features-in-scikit-learn) 使用 sklearn的 FutureUnion 功能，或者直接ColumnTransformer，举个例子:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

X = df[['text1_column_name', 
        'text2_column_name', 
        'standard_feature1', 
        'standard_feature2']]
y = df['target']

model = RandomForestClassifier()
vectorizer1 = TfidfVectorizer()
vectorizer2 = TfidfVectorizer()

column_transformer = ColumnTransformer(
    [('tfidf1', vectorizer1, 'text1_column_name'), 
    ('tfidf2', vectorizer2, 'text2_column_name')],
    remainder='passthrough')

pipe = Pipeline([
                  ('tfidf', column_transformer),
                  ('classify', model)
                ])
pipe.fit(X,y)
```

通常情况下，如果我们已经将文本转成了稀疏矩阵（e.g., ）作为特征，当我们需要往这个矩阵里再被一些新的特征时，可以使用 [`scipy.sparse.stack`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.hstack.html) 将新的特征加到这个稀疏矩阵里。 好处是可以尽可能的减少内存占用。 


# References 
* [6.2. Feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
* [10+ Examples for Using CountVectorizer](https://kavita-ganesan.com/how-to-use-countvectorizer/)
* [CountVectorizer - sklearn document](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
* [TfidfTransformer - sklearn document](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer)