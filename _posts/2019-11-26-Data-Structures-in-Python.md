---
layout:     post
title:      Data Structures in Python
date:       2019-11-26
tags: [Python3, data structure]
categories: 
- Python3
- heap
- algorithm
---

# Heapq

Heap data structure is mainly used to represent a priority queue. 
The property of this data structure in Python is that each time the **smallest of heap element is popped(min heap)**. Whenever elements are pushed or popped, **heap structure in maintained**.

## Methods 
- `heapify(iterable)`, convert the iterable into a heap data structure
- `heappush(heap, element)`, insert `element` in to `heap`, and maintain the heap invariant 
- `heappop(heap)`, remove and return the smallest element
- `heappushpop(heap, item)`, combine the functioning of both push and pop in one statement, push first and pop later
- `heapreplace(heap, item)`, combine the functioning of both pop and push in one statement, pop first and push later
- `nlargest(k, iterable, key = fun)` return the k largest elements from the iterable specified and satisfying the 
key if mentioned.
- `nsmallest(k, iterable, key = fun)` return the k smallest elements from the iterable specified and satisfying the 


Examples: 
```python
import heapq
alist = [7, 3]
heap.heapify(alist) # converts list into heap 3, 7
heap.heappush(alist, 2) # pushes number 2.
head.nlargest(2, alist) # produces [7, 3]

heap = []
for item in [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]:
    heapq.heappush(heap, item)

ordered = []
while heap:
    ordered.append(heapq.heappop(heap) )
print(ordered)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## PriorityQueue Implementation 
Sometimes, it would be convenient to customize your own Priority Queue with different methods. 
A naive implementation of PQ can be found [here]({{site.baseurl}}/codes/pq.py.txt).

Note that, the complexity of `push` method of this PQ is $$O(n)$$ while `heapq.heappush` is $$O(\text{log}(n))$$. 
But the `pop` method is faster that `heapq.heappop`, since the former is $$O(1)$$ while the latter is $$O(\text{log}(n))$$. 

