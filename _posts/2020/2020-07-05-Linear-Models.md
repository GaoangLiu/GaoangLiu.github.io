---
layout: post
title: Linear Models
date: 2020-07-05
tags: linear logistic gradient_descent
categories: machine_learning
author: berrysleaf
---
* content
{:toc}


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

For this function, there is a *closed-form* solution, called **Normal equation**, $$\hat{\theta} = (X^T \cdot X)^{-1} \cdot X^T \cdot y$$. 

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
    * escape local minima 
* cons: 
    * algorithm is much less regular 
    * may never converge to the optimal parameter value. We can use *simulated annealing* method to reduce learning rate during training. 

---

# Regularized Linear Models
Regularization is unnecessary unless your model is over-fitting, but generally a little regularization won't hurt much on the performance. Regularization is based on the idea that: the less freedom the model has, the harder it will be for it to overfit the data. 

For linear models, regularization is typically achieved by **constraining the weights of the model**. 

## Ridge Regression 
Adds a regularization term $$ \alpha \sum_{i=1}^n \theta_i^2 $$ to the cost function. This forces the learning algorithm to not only fit the data but also keep the model weights as small as possible. 

Ridge regression cost function:
$$
\begin{aligned}
    J(\theta) = \text{MSE}(\theta) + \frac{\alpha}{2} \sum_{i=1}^n \theta_i^2 
\end{aligned}
$$

## Lasso Regression 
Full name: Least Absolute Shrinkage and Selection Operator Regression with a cost function
$$
\begin{aligned}
    J(\theta) = \text{MSE}(\theta) + \alpha \sum_{i=1}^n |\theta_i| 
\end{aligned} 
$$

The regularization item uses the $$l_1$$ norm of the weight vector instead of half the square of the $$l_2$$ norm.

An important characteristic of Lasso Regression is that it tends to completely eliminate the weights of the least important features (i.e., set them to zero). In other words, Lasso Regression automatically performs **feature selection** and outputs a **sparse model** (i.e., with few nonzero feature weights). 

<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='35px'> Question: why $$l_1$$ norm tends to produce a sparse model? 

<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='35px'> Answer: the subgradient vector of Lasso regression is $$ g(\theta, J) = \nabla_\theta\text{MSE}(\theta) + \alpha sign(\theta)$$, where $$sign(\theta) = (\frac{\theta_1}{|\theta_1|}, ..., \frac{\theta_n}{|\theta_n|})^T $$, which is always 1 or -1 except when $$\theta_i = 0$$. 
Therefore, the learning step $$- \eta \cdot g(\theta, J)$$ is kind of consistent at each iteration, and weights with smaller value will reach 0 early. 
check out [this answer on SO](https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models) for more detail. 


## Elastic Net
Elastic Net is simply a middle ground between Ridge Regression and Lasso Regression. The cost function is:
$$
\begin{aligned}
    J(\theta) = \text{MSE}(\theta) + r \alpha \sum_{i=1}^n |\theta_i| + \frac{1-r}{2} \alpha \sum_{i=1}^n \theta_i^2
\end{aligned} 
$$

<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='35px'> Question: when should you use Linear Regression, Ridge, Lasso, or Elastic Net? 

<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='35px'> Answer: 
* Ridge is a good default. It is almost always preferable to have at least a little bit of regularization, so generally you should avoid plain Linear Regression. 
* If you suspect that only a few features are actually useful, you should prefer Lasso or Elastic Net since they tend to reduce the useless features’ weights down to zero. 
* In general, Elastic Net is preferred over Lasso since Lasso may behave erratically when the number of features is greater than the number of training instances or when several features are strongly correlated.


# Logistic Regression 
Logistic Regression (also called *Logit Regression*) is commonly used to estimate the probability that an instance belongs to a particular class (e.g., what is the probability that this email is spam?). It can also be used for classification tasks. For example, to create a binary classifier, we can train a model that classifies an instance as positive if the estimated probability is greater than 50%, or negative otherwise. 


Logistic function 
$$
\begin{aligned}
\sigma(t) = \frac{1}{1 + \exp^{-t}}
\end{aligned}
$$
This function is *sigmoid* function, it takes the model result as input, and outputs a number between 0 and 1, aka the *logistic*.

## Estimated probability and cost function 
LR model estimated probability $$\hat{p} = h_\theta(x) = \sigma (\theta^T \cdot x)$$, and the model prediction $$ \hat{y} = int(\hat{p} \geq 0.5)$$. 

Cost function for a single instance: 
$$ c(\theta) = \begin{cases}
        -log(\hat{p}), & \text{ if } y = 1\\
        -log(1 - \hat{p}), & \text{ if } y = 0
        \end{cases}$$

The cost function over the whole training set is :

$$ J(\theta) = - \frac{1}{m} \sum_{i=1}^m [y^{(i)} log(1 - \hat{p}^{(i)}) + (1 - y^{(i)}) log(1 - \hat{p}^{(i)})]$$.

There is no closed-form equation for this function, but this function is convex, thus Gradient Descent (or any other optimization algorithm) is guaranteed to find the global minimum. 