---
layout:     post
title:      Fenwick Tree
date:       2020-04-13
tags: [Algorithm, fenwick tree, bit]
categories: 
- Algorithm
---

Fenwick tree, also called Binary Index Tree (BIT), is a data structure that can efficiently ($$O(log n)$$ time) update elements and calculate prefix sums in a table of numbers.

# Why BIT ?
Consider the following problem, given an array $$a = [1,3,5,2,8,6,5]$$, we want to:
1. Get the sum of the first $$i: i \le a.size()$$ elements
2. Update the value of a specified element

There are two straightforward solutions: 
1. Run a loop from 0 to $$i-1$$ element to calculate the sum, time $$O(n)$$; update a value by simply running $$a[i] += val$$, time $$O(1)$$; 
2. Store sum of the first $$i$$-th element into an extra array. Get the sum of a given range can be done in $$O(1)$$, but update a value and recompute the sum takes $$O(n)$$ 

Both solutions run in $$O(n)$$ time, which is not bad, but far from satisfiable. 
And here comes BIT, performing both update and query operations in $$O(log n)$$ time without consuming much extra memory.

# How ?

The idea is based on the fact that **all positive integers can be represented as the sum of powers of 2**. E.g., 13 = 8(2^3) + 4(2^2) + 1(2^0). Each node of BIT would store the sum of previous $$n$$ elements, where $$n$$ is a power of 2.

1. Assume the size of array is $$n$$, we will need extra $$(n+1)$$-sized array, initiated with zeros and written as $$T_{bit}$$;
2. Make sure for each $$i, T_{bit}[i] = \text{sum}(a[p(i) .. i))$$ (right side is exclusive), where $$p(i) = i - (i \& (-i))$$ is the parent node index of $$i$$. 
    * For example, when $$i = 4, p(i) = 0$$, then $$T_{bit}[4] = a[0]+a[1]+a[2]+a[3]$$. 
    * For another example, when $$i = 7, p(i) = 6$$, then $$T_{bit}[7] = a[6]$$. 

Step 2 actually relies on our `update(T, i, val)` procedure, which:
1. Adds the `val` to `T(i)`
2. Transmits the change to its ancestor node, first to its parent $$p[i] = i + (i \& (-i)$$, then update `i` to `p[i]`. 

That is `while i < n: T[i] += val and i = p[i]`;


# Implementation (Python3)
```python
class FenwickTree:
    def __init__(self, _size):
        self.tree = [0] * _size

    def prefix_sum(self, i):
        _sum = 0
        while i > 0:
            _sum += self.tree[i]
            i -= (i & (-i))
        return _sum

    def update(self, i, val):
        while i < len(self.tree):
            self.tree[i] += val
            i += (i & (-i))
```



# References 
* [Fenwick Tree - Wikipedia](https://en.wikipedia.org/wiki/Fenwick_tree)
* [BIT - geeksforgeeks](https://www.geeksforgeeks.org/binary-indexed-tree-or-fenwick-tree-2/)