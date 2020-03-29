---
layout:     post
title:      Rust Vectors 
date:       2020-03-24
img: 
tags: [rust, vector]
catagories: [programming]
---

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



# Reference 
1. [Rust Doc](https://doc.rust-lang.org/1.8.0/book/iterators.html)
