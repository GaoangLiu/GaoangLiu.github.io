---
layout:     post
title:      Algorithm 
date:       2019-06-23
author:     Gal
catalog: 	true
tags:
  - algorithm
---

## Golden Section Search(GSS)

GSS is the limit of [Fibonacci search](https://en.wikipedia.org/wiki/Golden-section_search). 通过不断缩小搜索空间，在一个单锋函数(unimodal function)寻找最值。

以下为寻找$$[a, b]$$间使得$$f(x) = (x -2 )^2$$最小的`x`的Python代码

### idea

假设 $$f(x)$$是一个二次函数，在区间$$[a,b]$$存在极小值(也必然是最小值，设为 $$x_m$$)，在$$[a,b]$$间选择两个点$$c, d (a < c < d < b)$$,显然有$$f(c) < \text{max}(f(a), f(d)), f(d) < \text{max}(f(a), f(d))$$ .

如果 $$f(c) < f(d)$$ ，then 必然有 $$x_m \in [a,d]$$ ($$x_m$$也可能位于$$a,c$$之间，取决于$$c$$的位置)。这时，用$$d$$代替$$b$$将缩小搜索区间。不断迭代这个过程，直到 $$abs(a-b) < eps \to 0$$。那么问题是：

>  如何选择 $$c, d$$ ?	 

GSS的策略是选择 $$c, d$$ 使得 $$(d-c) / (c-a) = (c-a) / (b-c)  = (d-c) /(b-d) $$ 如此，以保证 $$c$$即不会太接近于$$a$$，或者太接近于$$b$$. i.e., $$ c - a ==  b - d $$. 令 $$ x = c - a == b - d, y = d - c $$，那么 $$ y / x = x / (x + y) => x^2 = xy + y^2 => (x/y)^2 - (x/y) = 1$$

求解关系式: $$x/y = \frac{1+\sqrt{5}}{2} = 1.618033988… $$

 

```python
"""python progra m for golden section search. """
gr = (math.sqrt(5) + 1) / 2
def gss(f, a, b, tol=1e-5):
    '''
    golden section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b], example: f = lambda x: (x-2)**2
    '''
    c = b - (b - a) / gr
    d = a + (b - a) / gr 
    while abs(c - d) > eps: # eps = 10**-9
        if f(c) < f(d):
            b = d
        else:
            a = c

        # recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2
```



### Practice 

如何从一个“山形”列表arr中找出最大值index？

```python
def findMaxElementIndex(arr):
	i, j = 0, len(arr)-1
	def chooseC(l, r):
		return l + int(round(r-l) * 0.382))
	
	def chooseD(l, r):
		return l + int(round(r-l) * 0.618))
	
	x1, x2 = chooseC(i, j), chooseD(i,j)
	while x1 < x2:
		if arr[x1] < arr[x2]:
			l = x1
			x1 = x2
			x2 = chooseD(x1, r)
		else:
			r = x2
			x2 = x1
			x1 = chooseC(l, x2)
	return A.index(max(A[l:r+1]), l)
	# 列表非连接函数，最终x1,x2未必能刚好达到 x1 == x2，以上方法目的将搜索区间快速缩小到一个常数范围，然后暴力求解	
```

## Gray Code 

### What is gray code
A Gray code, also known as **reflected binary code(RBC)**, is an encoding of numbers so that adjacent numbers have a single digit differing by 1.

Given a number $$n$$, generating n-bit Gray codes is generating bit patterns from 0 to $$2^n-1$$ such that successive patterns differ by one bit. For example, when $$n = 2$$, a 2-bit sequence is: `00 01 10 11` (`00 10 11 01` is also valid). 


### How to generate n-bit Gray code
There is a pattern for it. Generally, $$n$$-bit Gray Codes can be generated from a list of $$(n-1)$$-bit Gray codes following:
1. Let $$L_{n-1}$$ be the list of $$(n-1)$$-bit Gray codes, we create another list $$L_n = L_a + L_b$$, where $$L_a = L_{n-1}, L_b = \text{reversed}(L_{n-1})$$
2. Modify the list $$L_a$$ by prefixing a `0` in all codes of it.
3. Modify the list $$L_b$$ by prefixing a `1` in all codes of it.
4. Let $$L_n = L_a + L_b$$. Then $$L_n$$ is the required list of $$n$$-bit Gray codes.

For example, we already know $$L_2=\{00, 01, 11, 10\}$$, then $$L_3 = \{000, 001, 011, 010\} \cup \{ 110, 111, 101, 100\}$$. 

<img class='center' src="{{site.baseurl}}/images/2019/graycode.png" width="40%">

Python code using recursion:
```python
def generate_gray_code(n):
    if n == 1: return ['0', '1']
    res = generate_gray_code(n - 1)
    return ['0' + i for i in res] + ['1' + i for i in res[::-1]]
```

A faster way to generate decimal gray code (i.e., in the form of $$\{0, 1, 3, 2\}$$) is :
```python
def generate_decimal_gray_code(n):
    return [i ^ (i >> 1) for i in range(1 << n)]
```



