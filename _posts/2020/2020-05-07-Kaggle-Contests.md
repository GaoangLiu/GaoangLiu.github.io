---
layout: post
title: Kaggle Contests
date: 2020-05-07
tags: kaggle contests
categories: machine_learning
author: GaoangLau
---
* content
{:toc}


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


## Bag of Words Meets Bags of Popcorn
wegpage: https://www.kaggle.com/c/word2vec-nlp-tutorial/submissions

CNN model: https://github.com/GaoangLiu/AA_ipynb/blob/master/Bag_of_Words_Meets_Bags_of_Popcorn_CNN.ipynb
Bert model: https://github.com/GaoangLiu/AA_ipynb/blob/master/Bag_of_Words_Meets_Bags_of_Popcorn_BERT.ipynb

Best ensemble result: `0.98027` (area under ROC curve), rank as 3rd if producible in 5 years ago. 

BERT results are generally better than `GloVe` + `CNN`(or `LSTM`). Best bert result is: `0.96441_bert_en_wwm_uncased_L-24_H-1024_A-16.csv`.

Some predictions:
<img src="https://i.loli.net/2020/05/26/1HVmjeNQPkZbI2K.png"; width=450; alt='bag of words predictions'>

