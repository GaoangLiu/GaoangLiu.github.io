---
layout: post
title: Understanding ownership, borrowing, lifetimes in Rust
date: 2020-03-20
tags: rust
categories: 
author: GaoangLiu
---
* content
{:toc}


First of all, Rust corner rules: 



1. Each value in Rust has a variable that’s called its **owner**.
2. There can **only be one owner at a time**.
3. When the owner goes out of scope, the value will be **dropped**.

## Ownership 
Ownership works differently from a garbage collector in other languages because it simply consists of **a set of rules** that the compiler needs to check at compile time. The compiler will not compile if ownership rules are not followed. The borrow checker is a compiler component that ensures the codes follows ownership. 

Understanding ownership requires an understanding of ‘stack’ vs ‘heap’ memory. In many, especially high-level, programming languages, the concepts of the stack and the heap are irrelevant to programmer. But in a systems programming language like Rust, whether a value is on the stack or the heap has more of an effect on how the language behaves and why you have to make certain decisions. 

### Stack/Heap Memory
The stack is memory that stores values in the order it gets them and removes the values in the opposite order. Data that is assigned to variables or passed as arguments to function calls are allocated onto the stack following the *last in, first out* specification.

The only type of data that can be stored on the stack is data that has a **known and fixed size**. Data with an unknown size at compile time or a size that might change must go onto the heap.

Manipulating data on heap memory is **slower**. To store new data on the heap, memory allocator finds an empty spot in the heap that is big enough, marks it as being in use, and returns a pointer, which is the address of that location. his process is called *allocating on the heap* and is sometimes abbreviated as just *allocating*.

> Rust primitive/scalar types (int, bool, float, char, string literal (NOT String) etc) are stored in stack memory.

### Ownership
Primitive types are popped from stack memory automatically when they go out of scope, while complex types must implement a [drop](https://doc.rust-lang.org/std/mem/fn.drop.html) function which Rust will call when out of scope (to explicitly deallocate the heap memory).

Here are some of the gotchas that trip people up:

- Primitive types are **copied** (because it’s cheap to copy stack memory).
- Primitive types have a **Copy trait** that enable this behavior.
- Complex types **move** ownership.
- Complex types do not have a **Copy trait**.

Take the following code as an example. By calling function `myprint`, the value assigned to variable `v` is moved to a new location (this process of transferring is also called **moving**), when you try to access variable `v`, you get `error[E0382]: use of moved value`, <img class='center' src="{{site.baseurl}}/images/2020/error.e0382_rounded.png" width="650">.

```rust
fn myprint(v: Vec<i32>){
        println!("{:?}", v);
}

fn main(){
        let v = vec![1,8,4,7];
        myprint(v); // ownership is transferred to function myprint
        myprint(v); // compile-time error
}
```
A solution here is to manually *copy* the value of `v` with method `v.clone()`, which duplicates the memory.

## Borrowing
The concept of borrowing is designed to **make dealing with ownership changes easier**. It does this by avoiding the moving of owners. Borrowing is like reference in C, it allows us to have multiple references to a resource.


## Reference 
1. [Understanding Rust: ownership, borrowing, lifetimes](https://medium.com/@bugaevc/understanding-rust-ownership-borrowing-lifetimes-ff9ee9f79a9c)
2. [Rust Means Never Having to Close a Socket](https://blog.skylight.io/rust-means-never-having-to-close-a-socket/)
3. [Rust doc](https://doc.rust-lang.org/1.8.0/book/ownership.html)










