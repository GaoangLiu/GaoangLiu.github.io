---
layout:     post
title:      Linear Models
date:       2020-07-05
tags: [linear, logistic, svm]
categories: 
- machine learning
---

# Linear Regression 

* Model prediction equation $$\hat{y} = \theta_0 + \sum_{i=1}^n \theta_i x_i$$, where 
    * $$\hat{y}$$ is the predicted value.
    * $$n$$ is the number of features.
    * $$x_i$$ is the `i` th feature value.
    * $$\theta_i$$ is the model `i` the parameter (feature weight).
    
* Above from can be vectorized into $$\hat{y} = \theta^T \cdot x$$, where 
    * $$\theta$$ is the model's *parameter vector*  (a column vector), and $$\theta^T$$ the transpose (a row vector). 
    * $$x$$ is the instance's *feature vector* (a column vector), containing $$x_0$$ to $$x_n$$, with $$x_0$$ always equal to 1. 

A common performance measure of a regression model is Mean Square Error (MSE), to train a model is therefore to tune the parameters to minimize the following function:
* MSE cost function for a Linear Regression Model: $$ \text{MSE} (\theta) = \frac{1}{m} \sum_{i=1}^m(\theta^T \cdot x^{(i)} - y^{(i)})^2$$

For this function, there is a *closed-form* solution, called * Normal equation, $$\hat{\theta} = (X^T \cdot X)^{-1} \cdot X^T \cdot y$$. 

The downside of normal equation is its high computational complexity. Typically, the complexity of inverting a matrix of size $$n \times n$$ is $$O(n^{2.4})$$ to $$O(n^3)$$

Another method to find the optimal solution is **Gradient Descent**. The general idea of this method is to **tweak parameters iteratively in order to minimize a cost function**. 

## Gradient Descent 
Mechanism: it measures the local gradient of the error function with regards to the parameter vector $$\theta$$, and goes in the direction of descending gradients. Once the gradient is zero, you've reached a minimum. 

Gradient vector of the cost function: 
$$
\begin{aligned}
    \nabla_\theta\text{MSE}(\theta) = \frac{2}{m} X^T \cdot (X \cdot \theta - y) 
\end{aligned}    
$$

At each step of iteration, parameter vector $$\theta$$ is updated $$\theta' = \theta - \eta \nabla_\theta \text{MSE}(\theta)$$. 

Be aware: 
* All features all standardized to similar scale before using GD, otherwise, the algorithm will take a long time to converge.  

The main problem with GD is the fact that the whole training set is used to compute the gradients at each step, which could be a problem when the training set is too large. A workaround is to use a (random) subset, or a single instance, to compute the gradient. 

## Stochastic Gradient Descent 
* pros: 
    * algorithm runs faster 
    * avoid local minima 
* cons: 
    * algorithm is much less regular 
    * may never converge to the optimal parameter value. We can use *simulated annealing* method to reduce learning rate during training. 



# Regularized Linear Models
Regularization is unnecessary unless your model is over-fitting the data, but generally a little regularization won't hurt much on the performance. The idea is: the less freedom the model has, the harder it will be for it to overfit the data. 

For linear models, regularization is typically achieved by **constraining the weights of the model**. 

## Ridge Regression 

