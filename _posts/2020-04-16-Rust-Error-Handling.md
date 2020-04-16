---
layout:     post
title:      Rust Error Handling
date:       2020-04-16
tags: [Rust, error handling]
categories: 
- Rust
---

# Error handling 
Two major categories of error: 
1. `recoverable`, errors such as *a file was not found*.  Rust has the type `Result<T, E>` for recoverable errors; 
2. ` unrecoverable`, errors, which are always symptoms of bugs, like trying to access a location beyond the end of an array. For this type of error, Rust has the `panic!` macro. 

## Handel unrecoverable error 



# The `?` operator for easier error handling

 RFC proposes a `?` operator in [PR #243: Trait-based exception handling](https://github.com/rust-lang/rfcs/pull/243) to make `error handling` easier and more concise.  
 `?` was introduced in Rust `1.22`, it is **a unary suffix operator which can be placed on an expression to unwrap the value on the left side of ? while propagating any error through an early return**: 
 ```rust
 let mut f = File::open("foo.txt")?; 
 ```
In this case, `?` applies to a `Result` value, and if it was an `Ok`, it unwraps it and gives the inner value. If it was an `Err`, it returns from the function we're currently in. This piece of code does the same thing as the following:

```rust
let f = File::open("username.txt");

let mut f = match f {
        Ok(file) => file,
        Err(e) => return Err(e),
};
```
 

# References 
* [The ? operator for easier error handling / Rust doc](https://doc.rust-lang.org/edition-guide/rust-2018/error-handling-and-panics/the-question-mark-operator-for-easier-error-handling.html)
* [Rust, the ? operator / Blog: https://m4rw3r.github.io/rust-questionmark-operator](https://m4rw3r.github.io/rust-questionmark-operator)
* [Error handling in Rust / Rust doc](https://doc.rust-lang.org/book/ch09-01-unrecoverable-errors-with-panic.html)
