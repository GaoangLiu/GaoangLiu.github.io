---
layout:     post
title:      Machine Learning Softwares
date:       2019-11-01
tags: [machine learning, keras, tensorflow]
categories: 
- machine learning
- papers
---

# TensorFlow 2.0 

## `tf.keras` V.S. `Keras`
Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.

To understand the relationship between `Keras` and `tf.keras`, first, we have to clarify the complicated, intertwined relationship between `Keras` and `TensorFlow`. 

1. 2015.03.27, [Francois Chollet](https://fchollet.com/)(who is also the author of Deep Learning with Python) committed and released the first version of Keras to his GitHub to facilitate his own research and experiments. 
2. Due to its easy-to-use API and the explosion of deep learning popularity, many developers, programmers, and machine learning practitioners flocked to Keras
3. Keras’ default backend was Theano (until v1.1.10)
4. Google released TensorFlow on November 9, 2015, Keras started supporting TensorFlow as a backend
5. Eventually, TensorFlow became the most popular backend, Keras v1.1.0 switched to TensorFlow as its default backend
6. The `tf.keras` submodule was introduced in TensorFlow v1.10.0, the first step in integrating `Keras` directly within the TensorFlow package itself.
7. Keras v2.3.0 was released on September 17, 2019. This is the final release of Keras that will support backends other than TensorFlow. 

To summary, `tf.keras` and `Keras` are two separated different modules. `Keras` is a high-level API of TensorFlow, and `tf.keras` is a submodule of TensorFlow.
It is recommended to use `tf.keras` for future projects as the Keras package will only support bug fixes.


# Keras 

