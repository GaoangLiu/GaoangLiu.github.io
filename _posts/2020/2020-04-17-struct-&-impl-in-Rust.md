---
layout: post
title: struct & impl in Rust
date: 2020-04-17
tags: rust impl struct
categories: rust
author: GaoangLiu
---
* content
{:toc}


# Structures 

3 types of structures ("structs") that can be created using the `struct` keyword:




- Tuple structs, which are, basically, named tuples.
- The classic [C structs](https://en.wikipedia.org/wiki/Struct_(C_programming_language))
- Unit structs, which are field-less, are useful for generics.

For example: 
```rust
struct Solution; # unit struct 
struct Pair(i32, String) # a tuple struct
struct Point { # a struct with two points 
  x: f32, 
  y: f32
}
```


At times when we do not particularly care what the default value of a struct field, we can use `#[derive(Default)]` 

```rust
#[derive(Default)]
struct Person {
    age: i32,
    info: HashMap(String, (i32, String)),
}

impl Person {
  fn new() -> Self{ Self::default() }
  ...
}
```

# References

* [Default trait - Rust Doc](https://doc.rust-lang.org/std/default/trait.Default.html)
