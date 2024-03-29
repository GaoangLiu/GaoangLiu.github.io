---
layout:     post
title:      iter v.s. into_iter in Rust
date:       2020-04-13
tags: [Rust, iterator]
categories: 
- Rust
---

The difference between `.iter()` and `.into_iter()` bugs me a lot.
In many cases, I have no idea which one is the right choice, and have to fight hard with the compiler to get my code to work. 

This is it. I have to figure it out today.

<img class='center' src="https://i.loli.net/2020/04/13/6Y8LVB9nSlqHsZ2.png"  alt="IMAGE ALT TEXT HERE" width="340">

So, what's the difference, anyway ? In short:
1. use `iter()` when you want to iterate the values by reference; 
2. use `into_iter()` when you want to **move**, instead of **borrow**, the value 

## About `iter()`
We can call `v.iter()` on something like a vector or slice. This creates an `Iter<'a, T>` type, which implements the Iterator trait and allows us to call functions like `.filter()`. 
`Iter<'a, T>` only has a reference to `T`, thus calling `v.iter()` will create a struct that *borrows* from `v`. 

Example, 
```rust
fn main() {
    let v = vec![1,5,3,7];
    let p = v.iter().filter(|a| **a > 3).collect::<Vec<_>>(); 
    println!("{:?} {:?}", p, v); // You can still access v here, since the ownership is not transferred
}
```

Note the double references `**a` here. `v.iter()` creates an iterator of references to elements. The closure passed to `filter()` takes another level of reference to the iterator item(find more detail on [Rust filter() doc](https://doc.rust-lang.org/std/iter/trait.Iterator.html)). 

However, we can destructure on the argument to strip away one:
```rust
let p = v.iter().filter(|&a| *a > 3).collect::<Vec<_>>();
```
or both:
```rust
let p = v.iter().filter(|&&a| a > 3).collect::<Vec<_>>();
```


## About `into_iter()`
This function creates a `IntoIter<T>` type that now has ownership of the original value. 
The word *into* is commonly used in Rust to signal that `T` is being `moved`.

This in the above example, if we replace the third line by:
```rust
let p = v.into_iter().filter(|a| *a > 3).collect::<Vec<_>>(); 
// we use into_iter()
```
The ownership of the value `v` is removed from `v` after the code, and we can no longer access (*borrow*) from it. 



# References 
* [Effectively using Iterator in Rust](https://hermanradtke.com/2015/06/22/effectively-using-iterators-in-rust.html)
