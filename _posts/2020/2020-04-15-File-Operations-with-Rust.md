---
layout: post
title: File Operations with Rust
date: 2020-04-15
tags: rust file
categories: rust
author: gaonagliu
---
* content
{:toc}


# Read File 

Files are automatically closed when they go out of scope.




```rust
use std::fs::File;
use std::io::prelude::*;

fn read_file() -> std::io::Result<String> {
    let mut file = File::open("foo.txt")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}
```
Note that the return type if `io::Result<String>` instead of `String`. 

Question: what is the question mark `?` for ?

A: Basically,  it unpacks the `Result` if `OK` and returns the `error` if not. It is equivalent to the `match` statement that we used to do error handling, this `?` makes code much more concise (if you do understand what error handling does). Refer [this](https://doc.rust-lang.org/edition-guide/rust-2018/error-handling-and-panics/the-question-mark-operator-for-easier-error-handling.html) for more details.

 Rewriting code with no question mark:
```rust
fn read_file_2 -> String {
    let mut file = File::open("foo.txt").expect("Unable to open file");
    let mut res = String::new(); 
    file.read_to_string(&mut res).unwrap(); 
    res 
}
```

A more efficient way to read the contents of a file is reading it with a [buffered Reader](https://doc.rust-lang.org/std/io/struct.BufReader.html).

```rust
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;

fn read_file(file_name: String) -> std::io::Result<String> {
    let mut file = File::open(file_name)?;
    let mut buf_reader = BufReader::new(file);
    let mut contents = String::new();
    buf_reader.read_to_string(&mut contents)?;
    Ok(contents)
}
```

# Write to file
```rust
fn write_to_file() -> std::io::Result<()> {
    let mut file = File::create("foo.txt")?;
    file.write_all(b"Hello, world!")?;
    Ok(())
}
```



# References 
* [File - Rust.doc](https://doc.rust-lang.org/std/fs/struct.File.html)
* [The ? operator for easier error handling](https://doc.rust-lang.org/edition-guide/rust-2018/error-handling-and-panics/the-question-mark-operator-for-easier-error-handling.html)
* [Question mark operator - stackoverflow](https://stackoverflow.com/questions/42917566/what-is-this-question-mark-operator-about)