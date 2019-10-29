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
  - Kaggle
---

# Data Science London + Scikit-learn

* [Contest Overview](https://www.kaggle.com/c/data-science-london-scikit-learn/overview)
* [ipynb code link](https://github.com/ssrzz/ssrzz.github.io/blob/master/codes/Kaggle_Data_Science_SKLearn.ipynb)

# Google Street View images identification 

Contest [Google Street View images identification ](https://www.kaggle.com/c/street-view-getting-started-with-julia/data)的目的在于推广Julia语言在ML中的应用。这里我们考虑仅使用`Python + DL`来解决问题。 

问题很简单，给出6220张 Google 街景照片(及labels)，要求对这个数据集进行拟合并预测测试集。 数据集中照片中包含52个英文字符(`[A-Za-z]`)以及10个数字(`0-9`)，即labels有62个类别。 

## 挑战
1. 图片规格不一，小的图片尺寸有 14 x 19(pixels)，大的有178 x 197
2. 数据集较小(6220)


## 方案 1： 简单 CNN
[last update: Oct-20-2019] 
部分代码：
```python
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(62, activation='softmax'))
    # opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model 
```

以上模型仅包含1个输入层，两个隐藏层，1个全连接层，一个输出层，迭代30次左右预测结果准确率 `0.725`.

## 方案 2： CNN + 数据增强
这种方案效果并不好，还不如不使用数据增强的效果。最好结果仅在0.68左右。 
思考了一下，可能原因在于：图片质量太差，有相当一部分图片尺寸很小而且模糊，人工识别都非常困难。 
ML中有一句名言："Garbage in, garbage out"。使用数据增强可能使模型更加关注于模糊的阴影与噪声，反而降低了准确率(存疑)。





# Colab

### Colab 挂载Google Drive方法

```bash
from google.colab import drive
drive.mount('/gdrive')
# list files
!ls /gdrive/'My Drive'/
```

第一次挂载需要验证绑定账号，挂载成功后在 `My Drive`中建文件夹、上传数据都可以通过 `/gdrive/'My Drive'`打开。 



