---
layout:     post
title:      Data Structures in Python
date:       2019-11-26
tags: [Python3, data structure, heapq, ordereddict]
categories: 
- Python3
- algorithm
---

# Ordereddict 
`from collections import ordereddict`

Methods:
1. `od.pop(k)`, remove a key 
2. `od.popitem()`, removes one pair from the dictionary as a tuple

Extended reading on StackOverflow Question [Are there any reasons not to use an OrderedDict?](https://stackoverflow.com/questions/18951143/are-there-any-reasons-not-to-use-an-ordereddict/18951209#18951209). 
The top answer given by Tim Peter (the guy who wrote `TIMSORT`) is, however, on why `OrderdedDict` is quite efficient.

## A simple implementation of Ordereddict with Dict + Double LinkedList
```python
class Node:
    def __init__(self, k, v):
        self.key = k
        self.val = v
        self.prev = None
        self.next = None

class MyOrderdedDict():
    def __init__(self):
        self.dic = dict()
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add(self, k, v):
        if k in self.dic:
            self.dic[k].val = v
        else:
            node = Node(k, v)
            self.dic[k] = node
            self.tail.prev.next, node.next = node, self.tail
            node.prev, self.tail.prev = self.tail.prev, node

    def _pop(self, k):
        if k not in self.dic:
            return
        node = self.dic[k]
        del self.dic[k]
        node.next.prev = node.prev
        node.prev.next = node.next

```

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
heapq.heapify(alist) # converts list into heap 3, 7
heapq.heappush(alist, 2) # pushes number 2.
headq.nlargest(2, alist) # produces [7, 3]

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


# Built-in methods
## `bisect`
### `bisect.insort`
`bisect.insort()` is an alias for `bisect.insort_right()`,  which is equivalent to `list.insert(bisect.bisect_right(list, item, lo, hi), item)`. This assumes that list is already sorted.

Time complexity $$O(n)$$, since the slow `insert` operation is dominating the time.

`bisect.insort_left()` does similar thing, but inserting item in list before any existing entries of item.

