---
layout:     post
title:      Colab Notebook - Online Notebook
date:       2020-04-15
tags: [colab, notebook]
categories: 
- machine learning
---

Official website: [Colab.research.google.com](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)

Features:
1. support GPU acceleration 
2. support LaTex formatting (in text cell)
3. Linux commands
4. and more


## Mount to Google Drive
```python
from google.colab import drive
drive.mount('/content/gdrive/')
```


## Manipulate images 
To display images in colab:
```python
from IPython.display import Image
Image('train/Train_0.jpg')
```


# Reference 
* [Overview of Colaboratory Features](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)
