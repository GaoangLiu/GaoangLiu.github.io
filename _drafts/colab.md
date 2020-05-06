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

## File transmission 
```python
from google.colab import files
files.download('example.txt') # This will download file from colab.research  

files.upload() # This will prompt user to select a file and upload it
```


## Mount to Google Drive
```python
from google.colab import drive
drive.mount('/content/gdrive/')
```


## Display images 
To display images in colab:
```python
from IPython.display import Image
Image('train/Train_0.jpg')
```

Another way to display image with `matplotlib`:
```python
from matplotlib import pyplot as plt
img = plt.imread('images/train_001.jpg')
print(img.shape)
plt.imshow(img)
```

# Colab specs
[Google Colab Hardware Specs](https://github.com/GaoangLiu/ipynb/blob/master/colab_system_specs.ipynb)
E.g., `GPU 0: Tesla T4 (UUID: GPU-d7c19abf-9992-15ea-19ff-7d7cccf652ac)`


# Reference 
* [Overview of Colaboratory Features](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)
