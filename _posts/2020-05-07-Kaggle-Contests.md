---
layout:     post
title:      Kaggle Contests
date:       2020-05-07
tags: [kaggle, contests]
categories: 
- machine learning
---

## Toxic Comment Classification Challenge 

Target: Identify and classify toxic online comments

* [Contest main page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview)
* [Solution ipynb file](https://github.com/GaoangLiu/ipynb/blob/master/Identify_and_classify_toxic_online_comments.ipynb)


## Quora Insincere Questions Classification 

Target: Build a model to **identify and flag insincere questions**, which could be disparaging, inflammatory or involve discrimination again certain groups of people. 

### Difficulties 
The data is highly imbalanced, sincere / insincere question ratio is 16:1. A naive classifier may focus on finding out sincere questions, resulting in a model with **high precision**, but **low recall**. This is also why in this contest, submissions are evaluated on [F1 Score](https://en.wikipedia.org/wiki/F1_score) between the predicted and the observed targets.

* [Contest main page](https://www.kaggle.com/c/quora-insincere-questions-classification/overview)
* [Solution ipynb file](https://github.com/GaoangLiu/ipynb/blob/master/Quora_Insincere_Questions_Classification.ipynb)
