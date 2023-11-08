---
layout: post
title: Process images with Python
date: 2020-04-21
tags: python image machine_learning
categories: python
author: berrysleaf
---
* content
{:toc}


# Image processing and decoding with `TensorFlow`



```python
# load the raw data from the file as a string
img = tf.io.read_file('m.jpg')
# convert the compressed string to a 3D uint8 tensor
img = tf.io.decode_jpeg(img, channels=3)
# Use `convert_image_dtype` to convert to floats in the [0,1] range.
img = tf.image.convert_image_dtype(img, tf.float32)
# resize the image to the desired size.
return tf.image.resize(img, [320, 320])
```


# References 

