---
layout: post
title: BinaryHeap in Rust
date: 2020-04-14
tags: rust binary_heap
categories: rust
author: gaonagliu
---
* content
{:toc}


A **priority queue** implemented with a binary heap (max-heap).




## Create a new queue 
```rust
use std::collections::BinaryHeap; 
let mut bh = BinaryHeap::new(); 
// or create from a vector 
let mut heap = BinaryHeap::from(vec![1, 2, 4, 5, 7]);
```

## Methods
* `.push()`, time complexity $$O(1)$$. The expected cost of `push`, averaged over every possible ordering of the elements being pushed, and over a sufficiently large number of pushes, is $$O(1)$$. The worst case cost of a single call to push is $$O(n)$$, which occurs when capacity is exhausted and needs a resize.
* `.pop()`,  time complexity $$O(log n)$$
* `.peek()/.peek_mut()`, time complexity $$O(1)$$
* `.len()`,  time complexity $$O(1)$$. Why not $$O(n)$$ ? Based on the [implemented code](https://github.com/rust-lang/rust/blob/master/src/liballoc/collections/binary_heap.rs#L885):
```rust
#[stable(feature = "rust1", since = "1.0.0")]
pub fn len(&self) -> usize {
    self.data.len()
}
```
The `.len()` method invokes `self.data.len()`, where `self.data` is the underlying vector, and the time complexity of vector `.len()` is $$O(1)$$ by, once again, checking on the [implemented code](https://github.com/rust-lang/rust/blob/1.25.0/src/liballoc/vec.rs#L1163-L1165):
```rust
pub fn len(&self) -> usize {
    self.len
}
```
* `.is_empty()`,    time complexity $$O(1)$$

### pop
`pub fn pop(&mut self) -> Option<T>`
Removes the greatest item from the binary heap and **returns it** (this is different from C++ `pop_heap()`, which doesn't return anything), or `None` if it is empty.


### Convert queue into vector 
`pub fn into_sorted_vec(self) -> Vec<T>`
Consumes the BinaryHeap and returns a vector in sorted (ascending) order.
```rust
let mut heap = BinaryHeap::from(vec![1, 2, 4, 5, 7]);
heap.push(6);
heap.push(3);

let vec = heap.into_sorted_vec();
assert_eq!(vec, [1, 2, 3, 4, 5, 6, 7]);
```
There is also a method `.into_vec()`: this consumes the BinaryHeap and returns the underlying vector in arbitrary order.

### Examples
An example of using `BinaryHeap` to solve the Leetcode problem [*1508. Range Sum of Sorted Subarray Sums*](https://leetcode.com/problems/range-sum-of-sorted-subarray-sums/description/) can be found [here](https://leetcode.com/problems/range-sum-of-sorted-subarray-sums/discuss/731873/rust-solution-with-binaryheap).


# References 
* [BinaryHeap - rust-lang.doc](https://doc.rust-lang.org/std/collections/struct.BinaryHeap.html)
* [BinaryHeap - Github source code](https://github.com/rust-lang/rust/blob/master/src/liballoc/collections/binary_heap.rs)
