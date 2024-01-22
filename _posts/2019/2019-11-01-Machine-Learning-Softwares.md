---
layout: post
title: Machine Learning Softwares
date: 2019-11-01
tags: machine_learning keras tensorflow seaborn
categories: machine_learning tools
author: GaoangLiu
---
* content
{:toc}

# Seaborn 

Question 1: why [Seaborn](https://seaborn.pydata.org/index.html) when we already have [Matplotlib](https://matplotlib.org) ?




Seaborn provides a high-level interface to Matplotlib, a powerful but sometimes unwieldy Python visualization library. As put by the official website:
> If matplotlib “tries to make easy things easy and hard things possible”, seaborn tries to make a well-defined set of hard things easy too.

## Functions 
Making a scatter plot is just one line of code using the `lmplot()` function.
```python
import seaborn as sns 
sns.lmplot(x='Attack', y='Defense', data=df)
sns.set_style('whitegrid') # Set theme
sns.violinplot(x='Type 1', y='Attack', data=df) # Violin plot
```


# TensorFlow 2.0 

## `tf.keras` V.S. `Keras`
Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.

To understand the relationship between `Keras` and `tf.keras`, we have to clarify the complicated, intertwined relationship between `Keras` and `TensorFlow`. 

1. 2015.03.27, [Francois Chollet](https://fchollet.com/) (i.e., the author of Deep Learning with Python) committed and released the first version of Keras to his [GitHub](https://github.com/fchollet) to facilitate his own research and experiments. 
2. Due to its easy-to-use API and the explosion of deep learning popularity, many developers, programmers, and machine learning practitioners flocked to Keras
3. At the beginning, the default backend of `Keras` was Theano (until v1.1.10)
4. Google released TensorFlow on November 9, 2015. Keras started supporting TensorFlow as a backend
5. Eventually, TensorFlow became the most popular backend, Keras v1.1.0 switched to TensorFlow as its default backend
6. The `tf.keras` submodule was introduced in TensorFlow v1.10.0, the first step in integrating `Keras` directly within the TensorFlow package itself.
7. Keras v2.3.0 was released on September 17, 2019. This is the final release of Keras that will support backend other than TensorFlow. Bugs present in multi-backend Keras will only be fixed until April 2020. 

To summary, `tf.keras` and `Keras` are two different modules. 

`Keras` is a high-level API, which at first has nothing to do with TensorFlow, but as the popularity of TF grows, `Keras` supported and switched to TF as the default backend. And now, it seems that TF will dominate the future of machine learning, `Keras` is integrated into TF. 

Tips: It is recommended to use `tf.keras` for future projects as the Keras package will only support bug fixes.

## How to update to `TensorFlow 2.0`
First of all, a virtual environment is strongly recommended to avoid potential package conflicts. 
```bash
virtualenv --system-site-packages -p python3 myenv
```
By running the above command, a virtual environment `myenv` is created. 
* `--system-site-packages` allows the projects within the virtual environment `myenv` access the global site-packages. The default setting does not allow this access.
* `-p python3` is used to set the Python interpreter.
* `myenv` is the name of the virtual environment we created

```bash
source /myenv/bin/activate
pip install --upgrade tensorflow==2.0.0-rc1
```

The above command installs a 2.0.0-rc1 CPU-only version.

To choose the appropriate TensorFlow version, visit [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip)

Alternative TensorFlow packages: 

* `tensorflow==2.0.0-rc1` Preview TF 2.0 RC build for CPU-only (recommended).
* `tensorflow-gpu==2.0.0-rc1` Preview TF 2.0 RC build with GPU support
* `tensorflow` Latest stable release for CPU-only.
* `tensorflow-gpu` Latest stable release with GPU support.
* `tf-nightly` Preview nightly build for CPU-only.
* `tf-nightly-gpu` Preview nightly build with GPU support.

### Test the installation 
```Python
import tensorflow as tf
print(tf.__version__)
```

### TensorFlow 2.0 Colab
[Google Colab](https://colab.research.google.com) is promoting TF 2.0 (current version is still TF 1.5, Nov 14, 2019), if you want to use TF2.0 on Colab, you can manually install it :
```python
!pip install tensorflow-gpu # or tensorflow-gpu==2.0.0-rc1
```
Note: you will have to install one of those packages with GPU support, otherwise there is no GPU acceleration even if you set the runtime type to GPU mode.


# Keras 
## Callbacks ?
A callback provides a set of functions to be applied at given stages of the training procedure.

### EarlyStopping 
Stop training when a monitored quantity has stopped improving.
```python
keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
``` 
Arguments:
- `monitor`: quantity to be monitored 
- `min_delta`: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
- `patience`: number of epochs that produced the monitored quantity with no improvement after which training will be stopped. Validation quantities may not be produced for every epoch, if the validation frequency (model.fit(validation_freq=5)) is greater than one.
- `baseline`: Baseline value for the monitored quantity to reach. Training will stop if the model doesn't show improvement over the baseline.
- `restore_best_weights`: whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used.
- `mode`: one of {auto, min, max}. In `min` mode, training will stop when the quantity monitored has stopped decreasing; in `max` mode it will stop when the quantity monitored has stopped increasing; in `auto` mode, the direction is automatically inferred from the name of the monitored quantity.

### ReduceLROnPlateau 
**Reduce learning rate** when a metric has stopped improving. For example, if `val_loss` stayed unreduced in 10 epochs, the learning rate is reduced by 90%, i.e., new_lr = lr * factor. 

```python
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```

## Sklearn
### LabelEncoder
Encode labels with value between 0 and n_classes - 1.
```python
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
r = le.fit_transform(["apple", "pen", "apple", "applepen"])
print(r) # [0, 2, 0, 1]
print(le.classes_) # ['apple', 'pen', 'applepen']
```
### LabelBinarizer
Very similar to `LabelEncoder`, but creating a label indicator matrix, instead an array, from a list of multi-class labels. 

```python
lb = sklearn.preprocessing.LabelBinarizer()
print(lb.fit_transform(['female', 'male', 'others', 'female'])) 
# [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
```

### OneHotEncoder
Encode categorical integer features as a one-hot numeric array. The input to this transformer must be an array-like of integers or strings, denoting the values taken on by categorical (discrete) features. 
```python
enc = OneHotEncoder(handle_unknown='ignore')
X = np.array(['female', 'male', 'others']).reshape(-1, 1)
print(enc.fit_transform(X).toarray())
# [[1. 0. 0.], [0. 1. 0.], [0. 0. 1.]]
```





