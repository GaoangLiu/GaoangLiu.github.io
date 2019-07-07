---
layout:     post
title:      Kaggle Competitions 
subtitle:   笔记与总结
date:       2019-07-05
author:     ssrzz
catalog: 	true
tags:
  - machine learning
  - CNN
  - TensorFlow
  - Keras
---

## MNIST - Digit Recognizer

MNIST ("Modified National Institute of Standards and Technology") — ML领域的“HELLO WORLD”. 

[Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/leaderboard)

* Train set 42000, label 为数字，图片以像素向量的形式给出($$[pixel_0, … , pixel_{783}]$$) 
* Test set 28000。目的：尽可能提高图片中数字识别率

### 随机森林

直接调用`sklearn`中RandomForest分类器进行学习验证。

```python
from sklearn.ensemble import RandomForestClassifier
def forest():
	forest_clf = RandomForestClassifier(random_state = 42, n_estimators=100)
	forest_clf.fit(x_train, y_train)
	test = pd.read_csv('test.csv')
	final_pred = forest_clf.predict(test)
```

结果：0.96542, 大约排在2400+/3200+的位置，

### CNN

卷积神经网络 2 层（1层32filters， 1层 64 filters) 

Echos = 6, train:test ratio 19:2(3800:400), result : 0.98528 (1630 / 3200+)

Echos = 16, train:test ratio 20: 1, result : 0.98642 基本上没有多大提升



## Colab

### Colab 挂载Google Drive方法

```bash
from google.colab import drive
drive.mount('/gdrive')
# list files
!ls /gdrive/'My Drive'/
```

第一次挂载需要验证绑定账号，挂载成功后在 `My Drive`中建文件夹、上传数据都可以通过 `/gdrive/'My Drive'`打开。 



