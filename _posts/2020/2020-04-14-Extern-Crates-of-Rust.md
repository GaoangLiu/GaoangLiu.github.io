---
layout: post
title: Extern Crates of Rust
date: 2020-04-14
tags: rust extern_crates
categories: rust
author: GaoangLau
---
* content
{:toc}


# Random 
Generate random numbers value using the thread-local random number generator.



First, we need to add `rand` to the dependencies in our project's Cargo.toml (should checkout [doc](https://rust-lang-nursery.github.io/rust-cookbook/algorithms/randomness.html) for the latest rand version).

```bash
[dependencies]
rand = "0.6"
```

## Usages 
```rust
let rand_u8 = rand::random::<u8>();
let rand_f64 = rand::random::<f64>();
if rand::random() { // generates a boolean
    println!("Better lucky than good!");
}
```
When using random in a loop, caching the generator as in the following example can increase performance.
```rust
use rand::Rng;
let mut rng = rand::thread_rng();
for x in 1..1000{
    let r_bool = rng.gen(); // which is faster than rand::random();
    let r_u32 = rng.gen::<u23>();
}
```

Use `gen_range(left, right)` to generates a random value within half-open `[left, right)` range (not including right).

```rust
let r_i32 = rng.gen_range(1, 100);
```


# References 
* [rand - Rust.doc](https://docs.rs/rand/0.7.2/rand/fn.random.html)
* [Generator random numbers - Rust.cookbook](https://rust-lang-nursery.github.io/rust-cookbook/algorithms/randomness.html)
