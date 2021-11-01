---
layout: post
title: Rust Directory File
date: 2021-10-17
tags: directory file
categories: rust
author: GaoangLau
---
* content
{:toc}


# Files
## Read
### How a read file contents



Method one, read file contents to a string with [`std::fs`](https://doc.rust-lang.org/rust-by-example/std_misc/file/open.html).
```rust
use std::{env, fs};
fn main() {
    let contents = fs::read_to_string(filename)
        .expect("Something went wrong reading the file");
}
```

Method two, read file as a `Vec[u8]` with [`read`](https://riptutorial.com/rust/example/9667/read-a-file-as-a-vec).
```rust
use std::fs;
fn main() {
    let data = fs::read("/etc/hosts").expect("Unable to read file");
    println!("{}", data.len());
}
```

Method three, read a file with `std::fs::File` and `std::io::Read`. Slightly more verbose but also more powerfull in that you can reuse allocated data or append to an existing object.

```rust
use std::fs::File;
use std::io::Read;

fn main() {
    let mut data = String::new();
    let mut f = File::open("/etc/hosts").expect("Unable to open file");
    f.read_to_string(&mut data).expect("Unable to read string");
    println!("{}", data);
}
```
Or, read a file as a `Vec[u8]`.
```rust
use std::fs::File;
use std::io::Read;

fn main() {
    let mut data = Vec::new();
    let mut f = File::open("/etc/hosts").expect("Unable to open file");
    f.read_to_end(&mut data).expect("Unable to read string");
    println!("{}", data);
}
```
Note, the method [`read_to_end`](https://doc.rust-lang.org/stable/std/io/trait.Read.html#method.read_to_end) we used here will read all bytes until EOF in this source, placing them into `buf`.

### What about Buffered/IO?
A buffered reader (or writer) uses a **buffer to reduce the number of I/O requests**. For example, it's much more efficient to access the disk once to read 256 bytes instead of accessing the disk 256 times.

An example using [`BufReader`](https://doc.rust-lang.org/std/io/struct.BufReader.html) on reading a file: 
```rust
use std::fs::File;
use std::io::{BufReader, Read};

fn main() {
    let mut data = String::new();
    let f = File::open("/etc/hosts").expect("Unable to open file");
    let mut br = BufReader::new(f);
    br.read_to_string(&mut data).expect("Unable to read string");
    println!("{}", data);
}
```

By the [document](https://doc.rust-lang.org/std/io/struct.BufReader.html), `BufReader<R>`  **improve the speed** of programs that **make small and repeated read calls** to the same file or network socket. That being said, `BufReader<R>` does not make much difference between [`read_to_end`]((https://doc.rust-lang.org/stable/std/io/trait.Read.html#method.read_to_end)) on reading a file, since `read_to_end` will read all bytes into memory within very few I/O requests.

This struct `BufReader` is typically useful when you want to read line-by-line. As [documented](https://doc.rust-lang.org/std/io/trait.BufRead.html) in rust doc:
> A BufRead is a type of `Reader` which has an internal buffer, allowing it to perform extra ways of reading.
> For example, reading line-by-line is inefficient without using a buffer, so if you want to read by line, you’ll need `BufRead`, which includes a `read_line` method as well as a `lines` iterator.



An example:
```rust
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let f = File::open("/etc/hosts").expect("Unable to open file");
    let f = BufReader::new(f);
    for line in f.lines() {
        let line = line.expect("Unable to read line");
        println!("Line: {}", line);
    }
}
```


<img src="https://i.loli.net/2021/10/19/MAsFdY4Ro3Pf6gL.png" height="800px">



## Write
### How to write contents to a file
Method 1, creates a new file and write bytes to it with [`write()`](https://doc.rust-lang.org/std/fs/struct.File.html)
```rust
use std::fs::File;
use std::io::prelude::*;
fn main() -> std::io::Result<()> {
    let mut file = File::create("foo.txt")?;
    file.write_all(b"Hello, world!")?;
    Ok(())
}
```
Write string to a file:
```rust
use std::fs;
fn main() {
    let data = "Some data!";
    fs::write("/tmp/foo", data).expect("Unable to write file"); 
    // or 
    let mut f = File::create("/tmp/foo").expect("Unable to create file");
    f.write_all(data.as_bytes()).expect("Unable to write");
}
```

Method 2, write with [`BufWriter`](https://doc.rust-lang.org/std/io/struct.BufWriter.html):
```rust
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let data = "Some data!";
    let f = File::create("/tmp/foo").expect("Unable to create file");
    let mut f = BufWriter::new(f);
    f.write_all(data.as_bytes()).expect("Unable to write data");
}
```

# Paths
<img src="https://i.loli.net/2021/10/20/ZDJ5oK9yRQOa4s6.png" width="800px">

## Module `std::path`
This module provides two types, [`PathBuf`](https://doc.rust-lang.org/std/path/struct.PathBuf.html) and [`Path`](https://doc.rust-lang.org/std/path/struct.Path.html) (akin to `String` and `str`), for working with paths abstractly. These types are thin wrappers around [`OsString`](https://doc.rust-lang.org/std/ffi/struct.OsString.html) and [`OsStr`](https://doc.rust-lang.org/std/ffi/struct.OsStr.html) respectively, meaning that they work directly on strings according to the local platform’s path syntax.

This type supports a number of operations for inspecting a path, including breaking the path into its components, extracting the file name, determining whether the path is absolute, and so on.


### `std::path::PathBuf`
A [`PathBuf`](https://doc.rust-lang.org/std/path/struct.PathBuf.html) is like `String` - it owns a growable set of characters, but with methods specialized to building up paths. Most of its functionality however comes from the borrowed version `Path`, which is like `&str`. So, for instance, `is_dir` is a Path method.


The difference between `Path` and `PathBuf` is roughly the same as the one between `&str` and `String` or `&[]` and `Vec`, ie. `Path` only holds a reference to the path string data but doesn't own this data, while `PathBuf` owns the string data itself. This means that a `Path` is immutable and can't be used longer than the actual data (held somewhere else) is available.

The reason why both types exists is to avoid allocations where possible, however, since most functions take both Path and PathBuf as arguments (by using `AsRef<Path>` for example), this usually doesn't have a big impact on your code.

A very rough guide for when to use Path vs. PathBuf:
- For return types: if the function gets passed a `Path[Buf]` and returns a subpath of it, you can just return a `Path` (like Path[Buf].parent()), if you create a new path, or combine paths or anything like that, you need to return a `PathBuf`.

- For arguments: Take a PathBuf if you need to store it somewhere, and a Path otherwise.

- For arguments (advanced): In public interfaces, you usually don't want to use Path or PathBuf directly, but rather a generic P: `AsRef<Path>` or `P: Into<PathBuf>`. That way the caller can pass in Path, PathBuf, &str or String. 

Example:
```rust
use std::path::PathBuf;
let mut path=PathBuf::new();
path.push("/tmp");

let another_path: PathBuf = [r"C:\", "windows", "system32.dll"].iter().collect();
```

### `std::path::Path`
A slice of a path (akin to `str`).  Usage:
- Create a `Path` slice from `str` slice:
```rust
let path = Path::new("/etc/hosts"); 

```


# References
- [rust gentle into filesystem](https://stevedonovan.github.io/rust-gentle-intro/3-filesystem.html)

