---
layout:     post
title:      Golden section search
subtitle:   求区间最小/大值
date:       2019-06-23
author:     ssrzz
catalog: 	true
tags:
  - algorithm
  - golden-section
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## GSS(golden section search)

GSS is the limit of [Fibonacci search](https://en.wikipedia.org/wiki/Golden-section_search). 通过不断缩小搜索空间，在一个单锋函数(unimodal function)寻找最值。

以下为寻找`[a, b]`间使得`f(x) = (x -2 )**2`最小的`x`的Python代码

### idea

假设 $f(x)$是一个二次函数，在区间`[a,b]`存在极小值(也必然是最小值，设为)，在`[a,b]`间选择两个点`c, d (a < c < d < b)`,显然有f(c) < max(f(a), f(d)), f(d) < max(f(a), f(d)) .

如果f(c) < f(d)  $x_2$

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

