---
layout:     post
title:      Rust Iterators
date:       2020-05-20
img: 
tags: [rust, iterator]
catagories: [programming]
---

[`std::iter::Iterator`](https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.chain)

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