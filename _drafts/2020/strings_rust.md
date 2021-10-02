---
layout:     post
title:      Strings in Rust
date:       2020-04-15
tags: [rust, string]
categories: 
- rust
---

<img src="https://cdn.jsdelivr.net/gh/ddots/stuff@master/2021/e5d33811-eef6-41bd-ad52-db575304d3ed.png" width="50%" height="300px">

Rust has only one string type in the core language, which is the **string slice** `str` that is usually seen in its borrowed form `&str`. String slices are **references** to some UTF-8 encoded string data stored elsewhere. String literals, for example, are stored in the binary output of the program and are therefore string slices.

The `String` type is provided in Rust’s standard library rather than coded into the core language and is a **growable, mutable, owned, UTF-8 encoded string type**. 

To summarize the differences, `String`, stored as a vector of bytes (`Vec<u8>`), is guaranteed to be a valid UTF-8 sequence. While, `&str`, is a slice (`&[u8]`) that always **points to** a valid UTF-8 sequence, and can be used to view into a `String`, just like `&[T]` is a view into `Vec<T>`.

# Create new String
1. use the `to_string` method
```rust
let s = "hello string".to_string();
```
2. use the function `String::from` to create a String from a string litera
```rust
let s = String::from("hello string");
```


# Indexing into String
Although we mentioned earlier that String types are stored as vectors of bytes, they are essentially [wrappers](https://docs.rs/wrapper/0.1.1/wrapper/) over a `Vec<u8>`. Direct indexing is not a good idea, to demonstrate, consider the following code:

```rust
let s = String::from("罄竹难书");
println!("{}", s.len()) ; // 12
```
Assume `&s[0]` is a valid operation in Rust, which it's not, users might expect a return value `罄` from `&s[0]`, but `s` has value `Bytes(Copied { it: Iter([231, 189, 132, 231, 171, 185, 233, 154, 190, 228, 185, 166]) })`. When encoded in UTF-8, each Chinese Unicode scalar value takes three bytes of storage, the first byte of `罄` is `231`, so `&s[0]` shoule be `231`, but `231` is not a valid character on its own. 

And even if the string contains only Latin letters, returning the byte value is probably not what users want, e.g., `&s[0]` returns `104` instead of `h` when `let s="hello";`.

## Make it EXPLICIT what you need
1. If you want a **string slice**, then use `[]` with a range to create a string slice containing particular bytes:
```rust
let s = String::from("罄竹难书");
println!("{}", &s[..3]) ; // 罄
```
Be careful when creating string slices, the program will crash if you mess up. Guess what will happen if we used `&s[..4]`? 

Rust will panic at runtime in the same way that accessing an invalid index in a vector does:
```bash
thread 'main' panicked at 'byte index 4 is not a char boundary; it is inside '竹' (bytes 3..6) of `罄竹难书`', src/main.rs:4:22
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

2. If you want to deal with Unicode scalar values, the best way to do so is to use the `chars` method:
```rust
let s = String::from("罄竹难书");
println!("{}", s.chars().nth(0).unwrap()); // or, 
println!("{}", s.chars().next().unwrap()); // or, 
```

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