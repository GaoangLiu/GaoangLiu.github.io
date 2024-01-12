---
layout: post
title: Algorithm
date: 2019-06-23
tags: algorithm
categories: 
author: GaoangLiu
---
* content
{:toc}


# 快速选择算法 QuickSelect
要解决的问题是：从无序数组中寻找第 $$k$$ 个最小的元素（最大的类似），在思路上跟快排比较类似（都是 Tony Hoare 提出的）。




快排的思路是：
1. 选择一个 pivot;
2. 将数组分为两部分，左边的元素都小于 pivot，右边的元素都大于 pivot；
3. 递归的对左右两部分进行快排。

Quick Select 也是递归的思路，差异在于，既然要确定第 $$k$$ 个最小的元素，只要确定一个 pivot，使得比它小的元素个数为 $$k-1$$ 个，至于这 $$k$$ 个元素之间的大小如何，这并不重要。因此，在做完一次 partition 后，只需要判断 pivot 的位置是否为 $$k$$。如果是，说明这个 pivot 就是第 $$k$$ 个最小的元素，如果不是，那么就继续对左边或者右边进行递归。

Quick Select 的平均时间复杂度为 $$O(n)$$，复杂度分析参考 [wikipedia](https://zh.wikipedia.org/wiki/%E5%BF%AB%E9%80%9F%E9%80%89%E6%8B%A9)。



# Bézout's identity
[Bézout's identity](https://en.wikipedia.org/wiki/B%C3%A9zout%27s_identity):
如果两个整数 $$a, b$$ 的最大公因子为 $$d$$，那么存在两个整数 $$x, y$$ 使用 $$ax+by=d$$。更一般的，对于任意整数 $$x, y$$，那么  $$ax+by$$ 都是 $$d$$ 的倍数。 

这样的一对整数 $$x, y$$ 称为 Bézout 系数，Bézout 系数可以通过[扩展欧几里得算法](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm)求出。而假设我们已经得到了这样一对系数 $$(x, y)$$，那么其他所有的系数都可以通过以下公式获得：

$$(x + k \frac{b}{d}, y - k \frac{a}{d})$$

举例， $$a = 5, b = 7$$，其中一对系数为 $$(x=3, y=-2)$$，那么其他任意系数的形式为 $$(3 + 7k, -2 - 5k), k \in N$$

## Extended Euclidean Algorithm
Python 代码实现
```python
def compute_coe(a, b):
    if a == 1 and b == 0:
        return 1, 0
    x, y = compute_coe(b, a % b)
    return y, x - (a // b) * y
```
解析： 不失一般性，假设 $$d = 1$$, 再假设 $$x, y$$ 是输入为 $$a, b$$ 时间的结果，即 $$ a*x + b * y = 1 $$，而 $$x_1, y_1$$ 是输入为 $$b, a % b$$ 时的结果。 代入公式 $$ a % b = a - (a/b) * b $$ 可得 $$ ay_1 + b(x_1 - (a/b) y_1)) = 1 $$，那么就有 $$ x = y_1, y = x_1 - (a/b) * y_1 $$ 。 

因为用到辗转相除，基本情况是 $$ a=1, b=0 $$ 。




# Golden Section Search(GSS)

GSS is the limit of [Fibonacci search](https://en.wikipedia.org/wiki/Golden-section_search). 通过不断缩小搜索空间，对一个单锋函数(unimodal function)寻找最值。

以下为寻找$$[a, b]$$间使得$$f(x) = (x -2 )^2$$最小的`x`的Python代码

## 思想

假设 $$f(x)$$ 是一个二次函数，在区间 $$[a,b]$$ 存在极小值(也必然是最小值，设为 $$x_m$$)，在 $$[a,b]$$ 间选择两个点 $$c, d (a < c < d < b)$$，显然有 $$f(c) < \text{max}(f(a), f(d))$$ 以及 $$ f(d) < \text{max}(f(a), f(d)) $$ 。

现在假设 $$f(c) < f(d)$$， 那么必然有 $$x_m \in [a,d]$$ ($$x_m$$ 也可能位于 $$a,c$$ 之间，这取决于 $$c$$ 的位置)。这时，用 $$d$$ 代替 $$b$$ 将缩小搜索区间。不断迭代这个过程，直到 $$abs(a-b) < eps \to 0$$。那么问题是：

>  如何选择 $$c, d$$ ?	 

GSS的策略是选择 $$c, d$$ 使得 $$ \frac{d-c}{c-a} = \frac{c-a}{b-c} = \frac{d-c}{b-d} $$。 如此以保证 $$c$$ 即不会太接近于 $$a$$，或者太接近于 $$b$$。 也即是, $$ c - a =  b - d $$. 令 $$ x = c - a, y = d - c $$，那么 $$ y / x = x / (x + y) => x^2 = xy + y^2 => (x/y)^2 - (x/y) = 1$$

求解关系式: $$x/y = \frac{1+\sqrt{5}}{2} = 1.618033988… $$

 

```python
"""python progra m for golden section search. """
gr = (math.sqrt(5) + 1) / 2
def gss(f, a, b, tol=1e-5):
    '''golden section search
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



## Practice 

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

# Gray Code 

## What is gray code
A Gray code, also known as **reflected binary code(RBC)**, is an encoding of numbers so that adjacent numbers have a single digit differing by 1.

Given a number $$n$$, generating n-bit Gray codes is generating bit patterns from 0 to $$2^n-1$$ such that successive patterns differ by one bit. For example, when $$n = 2$$, a 2-bit sequence is: `00 01 10 11` (`00 10 11 01` is also valid). 


## How to generate n-bit Gray code
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



# SORT 
## TimSort 
- 提出者: [Tim Peters](https://en.wikipedia.org/wiki/Tim_Peters_(software_engineer))
- 出现时间： 2002 
- 应用: 从 `Python 2.3` 开始作为 `Python` 的标准默认算法，e.g., `a_list.sort()`
- 复杂度
    - worst-case $$O(n \text{log}n)$$
    - best-case $$O(n)$$
    - average $$O(n \text{log}n)$$
    - worst-case space $$O(n)$$

[More info](https://bugs.python.org/file4451/timsort.txt)

<!-- [MI2](https://hackernoon.com/timsort-the-fastest-sorting-algorithm-youve-never-heard-of-36b28417f399) -->

### Implementation
算法结合了**合并排序**及**插入排序**。

```python
RUN = 32

# This function sorts array from left index to  
# to right index which is of size at most RUN  
def insertion_sort(arr, left, right):
    for i in range(left + 1, right + 1):
        j = i 
        while j > left and arr[j - 1] > arr[j]:
            arr[j-1], arr[j] = arr[j], arr[j-1]
            j -= 1

def merge_sort(arr, left, mid, right):
    pass

def tim_sort(arr):
    n = len(arr)
    for i in range(0, n, RUN):
        insertion_sort(arr, i, i + min(i + RUN - 1, n - 1))
    
    # start merging from size RUN (or 32). It will merge to form size 64, then 128, 256 and so on ....  
    size = RUN 
    while size < n:  
        # pick starting point of left sub array. We  
        # are going to merge arr[left..left+size-1]  
        # and arr[left+size, left+2*size-1]  
        # After every merge, we increase left by 2*size  
        for left in range(0, n, 2*size):  
            # find ending point of left sub array  
            # mid+1 is starting point of right sub array  
            mid = left + size - 1 
            right = min((left + 2*size - 1), (n-1))  
    
            # merge sub array arr[left.....mid] &  
            # arr[mid+1....right]  
            merge(arr, left, mid, right)  
          
        size = 2*size 

```


# Bit operation 
## Count set bits in an integer 
### Simple method: Loop
```python
count = 0
while n :
    count += n & 1
    n >> = 1
```

### Brian Kernighan’s Algorithm
```python
ct = 0
while n:
    n = n & (n-1)
    ct += 1
```
The key step `n & (n-1)` set the rightmost `1` to `0`, e.g., when`n = 3 = 101, n & (n-1) = 1`. 

