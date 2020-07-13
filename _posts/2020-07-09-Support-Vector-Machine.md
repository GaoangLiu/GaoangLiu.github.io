---
layout:     post
title:      Support Vector Machine
date:       2020-07-09
tags: [svm]
categories: 
- machine learning
---
TODO: Translate to CN. 

<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='25px'> What is SVM exactly ?

<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='25px'> 
* The fundamental idea behind SVM is to **fit the widest possible 'stree' between the classes**. It other words, the goal is to have the largest possible margin between the decision boundary that separates the two classes and the training instances.
* It is capable of performing linear or nonlinear **classification, regression, and even outlier detection**. It is one of the most popular models in Machine Learning.
* SVMS are particularly well suited for classification of complex but small- or medium-sized datasets.


--- 

<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='25px'> What is support vector ?
<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='25px'>  Support vector are instances that determine the *decision boundary*, a line classifying instances into different categories. 

TODO: support vector determines decision boundary or the other way around ?

--- 

<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='25px'> Pros and Cons ?

<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='25px'> Pros:
* xx

Cons:
* 对特征缩放敏感。因为 SVM 试图最大化类别之前的间隔，如果训练集没有进行特征缩放，那么 SVM 会倾向于忽略数据值较小的特征.


--- 

<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='25px'> Capable of multi-class classification ?

<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='25px'> Yes

--- 

<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='25px'> What is kernel trick ? How it works?

<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='25px'> 

--- 

<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='25px'> What is similarity function ? Difference with kernel trick ?

<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='25px'> 

--- 

