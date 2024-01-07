---
layout: post
title: Rust Vectors
date: 2020-03-24
tags: rust vector
categories: 
author: gaoangliu
---
* content
{:toc}

## Initiation

Initiated from a range. 



```rust
let v_int:Vec<i32> = (1..10).map(i32::from).collect(); // or 
let v_int:Vec<_> = (1..10).map(f32::from).collect();
```
Where `i32::from` is an implementor of trait [`std::convert::from`](https://doc.rust-lang.org/std/convert/trait.From.html) to do value-to-value conversions while consuming the input value. E.g., `assert_eq!(i32::from(true), 1)`. 



## Iterators 
There are several ways to loop through a given vector `arr`, for example:
```rust
for i in 0..arr.len() {
    println!("{}", arr[i]);
}
```

This `C`-style loop is disencouraged by Rust, while the following iterate way is preferred: 
```rust
for n in &arr {
    println!("{}", n);
}
```
Benefits of the second way:

1. codes are much more clear and efficient, since we do not have to iterate through indexes, and then index the vector;
2. the first way will have extra bounds checking because it used indexing, `nums[i]`, while the second yields **reference** to each element and bound checking is not required. 

Two details we should notice: 
1. in the second version, we want to print `n: i32`, but the type `n` in the code is `&i32`. This is not a problem, sinc `println` handles the dereferencing for us;

2. we're using a reference to `arr`, i.e., `&arr`, this is because if we use the data itself, we will have to be its owner, which would involve making a copy of the data and giving us the copy.



## Sort

To sort a vector, use `vec.sort();`. This default `sort` method on slices is stable and $$O(nlogn)$$  worst-case. 
By stable, we mean the original order of equal elements is preserved after sorting is guaranteed.


If stability is not required, there is an alternative `sort_unstable`, which is faster and allocates no auxiliary memory.


## Query an element
* To get the first (leftmost) index of an element, use [`position`](https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.position) method:
```rust
let v = vec![1,3,5,3,7];
assert_eq!(v.iter().position(|&x| x == 3), Some(1));
```
* To get the last (rightmost) index of an element, use [`rposition`](https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.rposition) method:
```rust
let v = vec![1,3,5,3,7];
assert_eq!(v.iter().rposition(|&x| x == 3), Some(3));
```
* Get the **max/min** element from `vec`: `*v.iter().max().unwrap()` or `*v.iter().min().unwrap()`. Note that there is a dereference symbol `*` before `v`.

* Find `max_by_key`. `let abs_max = *a.iter().max_by_key(|x| x.abs()).unwrap();`. This returns the element that gives the maximum value from the specified function, i.e., `.abs()`.

* `max_by`, returns the element that gives the maximum value with respect to the specified comparison function. E.g., `let v = *a.iter().max_by(|x, y| x.cmp(y)).unwrap();` 

* `min_by`, `min_by_key` runs in similar way. 


## Zip & Unzip 
To zip up two iterators into a single iterator of pairs. `.zip()` returns a new iterator that will iterate over two other iterators, returning a tuple where the first element comes from the first iterator, and the second element comes from the second iterator.

```rust
let a1 = [1_i32, 3, 5]; 
let a2 = [2_i32, 4, 6];
let mut zipped = a1.iter().zip(a2.iter());
println!("{:?}", zipped);
// Zip { a: Iter([1, 3, 5]), b: Iter([2, 4, 6]), index: 0, len: 3 }
println!("{:?}", zipped.next());
// Some((&1, &2))
```

Not that, if `a1.len() != a2.len()`, the resulted iterator will stop on the shortest length.




## Other Methods

* `vec.clear()`, remove all values, but has no effect on the allocated capacity of the vector; 
* `vec.dedup()`, removes consecutive repeated elements in the vector. If the vector is sorted, this will remove all duplicate elements. 
* `vec.binary_search(&n)`, return `Result::Ok(idx)` if the value is found, or `Result::Err(idx)` elsewise. To get the `usize` index of a number, use `vec.binary_search(&42).unwrap_or_else(|x| x)`


# Reference

1. [Rust Doc](https://doc.rust-lang.org/1.8.0/book/iterators.html)
2. [Rust vector Github source code](https://github.com/rust-lang/rust/blob/master/src/liballoc/vec.rs)
