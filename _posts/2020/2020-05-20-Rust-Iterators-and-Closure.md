---
layout: post
title: Rust Iterators and Closure
date: 2020-05-20
tags: rust iterator
categories: 
author: GaoangLau
---
* content
{:toc}


# Closures
Closures, also known as **lambda expressions**, are **functions that can capture the enclosing environment**. For example, a closure that captures the `x` variable: `|val| val + x`.




Characteristics of closure includes:
1. using `||` instead of `()` around input variables.
2. optional body delimitation `({})` for a single expression (mandatory otherwise).
3. the ability to capture the outer environment variables.

Demo 
```rust
let double = |i| { i * 2 }; 
// equal to let double = |i:i32|->i32 {i*2};
let z = double(10);
```
Binding to reference annotation, i.e., named function, is optional. Also the argument and return **types are deducted** automatically so we do not have to declare them explicitly in the code.


# Iterators

[`std::iter::Iterator`](https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.chain)

## `fold`
Folds every element into an accumulator by applying an operation, returning the final result.
`fold()` takes two arguments: **an initial value**, and **a closure** with two arguments: an accumulator, and an element. The closure returns the value that the accumulator should have for the next iteration.

### Example 1
```rust
let a = [1,2,3];
let sum = a.iter().fold(0, |acc, n| acc + n); // 6
```

### Example 2
```rust
// Calculate sum of distances all characters from 'a' in the following string.
let ss = "abcdefg";
let n = ss.as_bytes().iter().fold((0, 'a'), |(sum, prev), &c| {
    (sum + (prev as i32 - 'a' as i32).abs(), c)
}).0;
println!("{:?}", n); // 15
```


## `flat_map`
Creates an iterator that works like map, but flattens nested structure. 
You can think of `flat_map(f)` as the semantic equivalent of `mapping`, and then flattening as in `map(f).flatten()`.

```bash 
fn flat_map<U, F>(self, f: F) -> FlatMap<Self, U, F> where
    F: FnMut(Self::Item) -> U,
    U: IntoIterator, 
```

### Examples:
```rust
let words = ["alpha", "beta", "gamma"];
let merged: String = words.iter().flat_map(|w| w.chars()).collect();
assert_eq!(merged, "alphabetagamma");
```

## `chain`
Takes two iterators and creates a new iterator over both in sequence.

### Examples
```rust
let a1 = [1, 2, 3]; 
let a2 = [4, 5, 6];
let mut iter = a1.iter().chain(a2.iter())
```

Combine `flat_map` and `chain` to capitalize a string: 
```rust
let s = "rust is awesome".to_string();
let t = s.chars().take(1).flat_map(char::to_uppercase).chain(s.chars().skip(1)).collect::<String>();
```