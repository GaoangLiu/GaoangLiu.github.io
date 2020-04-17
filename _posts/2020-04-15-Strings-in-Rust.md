---
layout:     post
title:      Strings in Rust
date:       2020-04-15
tags: [rust, string]
categories: 
- rust
---

There are two types of `string` in Rust: 
1. `String`, stored as a vector of bytes (`Vec<u8>`), but guaranteed to always be a valid UTF-8 sequence. String is **heap allocated**, growable and not null terminated.
2. `&str`, is a slice (`&[u8]`) that always points to a valid UTF-8 sequence, and can be used to view into a `String`, just like `&[T]` is a view into `Vec<T>`.


# Conversion 
String <=> i32
```rust
let s = "123".to_sring(); 
let int_s = s.parse::<i32>().unwrap(); 
```

And to split a string into a vector:
```rust
let s = "12 34 56 87 65 90"; 
let mut cs = s.split(" ").map(|c| c.parse::<i32>().unwrap()).collect::<Vec<i32>>();
```

Lowercase & Uppercase

```rust
s.to_lowercase() 
s.to_uppercase() 
c.is_numeric() # To determine whether a char is number or not
```



# References

* [String - Rust.doc](https://doc.rust-lang.org/std/string/struct.String.html)struct.File.html)
* [Strings - Rust by example](https://doc.rust-lang.org/rust-by-example/std/str.html)