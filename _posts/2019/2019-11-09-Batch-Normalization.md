---
layout: post
title: Batch Normalization
date: 2019-11-09
tags: machine_learning
categories: machine_learning
author: GaoangLau
---
* content
{:toc}


## What is batch normalization 




## Why we use it
Generally speaking, when training a neural network, we want to normalize or standardize our data in pre-processing step to put all the data in the same scale.   
$$z = (x-m) / s $$

Without normalization, relatively large inputs can cascade down through the layers in the network, which may cause imbalance gradients, which may cause the famous **exploding gradient problem**.
Besides, non-normalized data can significantly decrease our training speed. 

But this is not the end of this *normalization story*, once the normalized input data were fed into the network, weights of the model were updated during each epoch via SGD. If one of those weights ends up becoming drastically larger than other weights, then the output from its corresponding neuron might be extremely large and this imbalance will again continue to cascade through the network causing instability. 

This is where BN comes into play. With BN, we have normalized data coming in and normalized data within the model.

- BN reduces the amount by what the hidden unit values shift around (covariance shift)
- BN allows each layer of a network to learn by itself a little bit more independently of other layers.
- It reduces overfitting because it has a slight regularization effects. Similar to dropout, it adds some noise to each hidden layer’s activations. 

## How to implement it in our algorithm 
Batch normalization normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation. 
This process occurs on per batch basis, hence the name **batch norm**.

## Alternatives 


## Materials 
* [Andrew Ng explains BN](https://www.youtube.com/watch?v=nUUqwaxLnWs)



