---
layout:     post
title:      Sorting Algorithms 
subtitle:   
date:       2019-07-05
author:     ssrzz
catalog: 	true
tags:
  - quick sort
  - merge sort
---

## Quick Sort

```py
def quicksort(arr, start=0, end=0):
    if start >= end:
        return
    pindex = partition(arr, start, end)
    quicksort(arr, start, pindex - 1)
    quicksort(arr, pindex + 1, end)

def partition(arr, start, end):
    pivot = arr[end]
    pindex = start
    
    # move all elements less than pivot to left of pindex
    for i in range(start, end):
        if arr[i] <= pivot:
            arr[i], arr[pindex] = arr[pindex], arr[i]
            pindex += 1
    arr[end], arr[pindex] = arr[pindex], arr[end]
    return pindex

```

* Inplace recursive 算法 
* Time complexity $$O(n  \cdot  logn)$$， worst-case running time $$O(n^2)$$



## Merge sort

* Not inplace 
* Time complexity $$O(n \cdot logn) $$, space complexity $$O(n)$$ 



# Bucket sort 

