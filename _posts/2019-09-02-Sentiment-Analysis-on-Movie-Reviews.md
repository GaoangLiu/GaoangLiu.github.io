---
layout:     post
title:      Sentiment Analysis on Movie Reviews
date:       2019-09-02
img: sentiment.png
tags: [movie reviews, sklearn, nltk]
---

# Sentiment Analysis on Movie Reviews
[Kaggle](http://www.kaggle.com/) 赛事之  [影评情感分析(Sentiment Analysis on Movie Reviews, SAMR)](http://www.kaggle.com/c/sentiment-analysis-on-movie-reviews).

基于：

* Python 3.7 
* `scikit-learn, nltk`

## 问题描述

摘自 Kaggle's [描述](http://www.kaggle.com/c/sentiment-analysis-on-movie-reviews):

本赛事考察参赛者对[烂番茄](http://www.rottentomatoes.com/)数据集进行情绪分析能力，之后参赛者需对每段评价做出预测，预测值为:[4,3,2,1,0]，分别对应 [positive, somewhat positive, neutral, somewhat negative, negative]中的一个。

示例：

* 4 (positive): "They works spectacularly well... A shiver-inducing, nerve-rattling ride" (好赞顶)
* 3 (somewhat positive): "rooted in a sincere performance by the title character undergoing midlife crisis"(剧本扎实、表演到位)
* 2 (neutral)："Its everything you would expect -- but nothing more." (该有的都有，但仅此而已)
* 1 (somewhat negative) ："But it does not leave you with much." （看完即忘)
* 0 (negative)："The movies progression into rambling incoherence gives new meaning to the phrase fatal script error."（烂的一踏糊涂）



## 思路-1

* 特征提取
  * 词频-逆向文档频率(term frequency-inverse document frquency)  
  * 删除停用词
  * 设置最小文档频率 (`min_df=5`)
* 逻辑回归
* 网格搜索

我们采用词袋来表示文本，词袋表示由`CounterVectorizer`实现。 

```python
from sklearn.feature_extraction.text import CounterVectorizer
vect = CountVectorizer().fit(text_train)
vect = CountVectorizer().fit(text_train)
x_train = vect.transform(text_train)

features = vect.get_feature_names()
print(len(features))  # 15240
print("First 20 features :\n{}".format(features[:20]))
# ['000', '10', '100', '101', '102', '103', '104', '105', '10th', '11', '110', '112', '12', '120', '127', '129', '12th', '13', '13th', '14']

scores = cross_val_score(LogisticRegression(solver='lbfgs'), x_train, y_train, cv=5)
print("Mean score ", np.mean(scores))
# 0.64	
```

训练集大小 156060 x 15240，表示成词表可以发现前20个元素都是数字，这些数字基本上不包含有用信息。我们采用设置最小文档频率 `min_df=5`(i.e., 仅使用至少在5个文档中出现过的词例)来约束这个词例。 

> `min_df=5` 特征数缩小到 14609，而交叉验证精度几乎没有变化

### 用tf-idf缩放数据

我们考虑使用词频-逆向文档频率来舍弃不重要的特征，同时对*在某一特定文档中经常出现的术语*给予很高的权重。之后使用网格搜索寻求最优参数。

最终提交结果得分：0.6073  排名 [414/861]

```python
pipe = make_pipeline(TfidfVectorizer(min_df=5, stop_words='english'), LogisticRegression(solver='lbfgs', multi_class='auto'))
param_id = {'logisticregression__C':[0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipe, param_id, cv=5)
grid.fit(text_train, y_train)
print("Best score {:.2f}".format(grid.best_score_))  # 0.63
```



### $$n$$-元分词

之前仅考虑了单一词例的计数，缺点在于这种方法完全舍弃了单词顺序与短语结构。比如，它无法区分 "not bad, this is a good movie" 与 "not good, this is a bad movie". 


```bash
Best cv score: 0.65
Best parameters :
{'logisticregression__C': 10, 'tfidfvectorizer__ngram_range': (1, 2)} 
```

Heatmap:

![img]({{site.baseurl}}/images/heatmap-ngram.png)


### 代码 
[code]({{site.baseurl}}/codes/Kaggle_Data_Science_SKLearn.ipynb)