---
layout: post
title: Rust Error Handling
date: 2020-04-16
tags: rust error_handling
categories: rust
author: GaoangLau
---
* content
{:toc}


# Error handling 
There are two major categories of error in Rust: 



1. `recoverable`, errors such as *a file was not found*.  Rust has the type `Result<T, E>` for recoverable errors; 
2. ` unrecoverable`, errors, which are always symptoms of bugs, like trying to access a location beyond the end of an array. For this type of error, Rust has the `panic!` macro. 

## Handle unrecoverable error with `panic!`
Two main usages: 
1. Explicitly call `panic!` in code when bad things happens; 
2. Use a `panic!` backtrace to debug.

For example 
```rust
fn main() {
        // explicitly invoke panic! method
        if dividend == 0 { panic!("Danger. Zero-valued dividend.")};
}
```

When the program crashed due to an error such as *buffer overread*, we can set the `RUST_BACKTRACE` environment variable to get a backtrace of exactly what happened to cause the error. E.g., `RUST_BACKTRACE=1 cargo run`. 

## Handle recoverable error 
There are several ways to do so:
```rust
// method 1, use match
let f = File::open("foo.txt"); 
let f = match f {
        Ok(file) => file, 
        Err(error) => {
                panic!("Problem opening the file: {:?}", error);
        }
}
``` 
This way of handling error generally works fine, but when there are multiple failure reasons, we want to take different action for each reason. This can be done with inner `match` expressions. 
```rust
let f = match f {
        Ok(file) => file,
        Err(error) => match error.kind() {
            ErrorKind::NotFound => match File::create("hello.txt") {
                Ok(fc) => fc,
                Err(e) => panic!("Problem creating the file: {:?}", e),
            },
            other_error => panic!("Problem opening the file: {:?}", other_error),
        },
    };
```
That's quit a log `match`, which is useful but also pretty primitive. To clean up huge nested `match` expression, we use `unwrap_or_else()` method:

```rust
let f = File::open("hello.txt").unwrap_or_else(|error| {
        if error.kind() == ErrorKind::NotFound {
            File::create("hello.txt").unwrap_or_else(|error| {
                panic!("Problem creating the file: {:?}", error);
            })
        } else {
            panic!("Problem opening the file: {:?}", error);
        }
});
```

TODO: [https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html)


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
