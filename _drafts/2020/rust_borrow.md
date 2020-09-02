---
layout:     post
title:      Understanding ownership, borrowing, lifetimes in Rust
date:       2020-03-20
img: 
tags: [rust]
catagories: [programming]
---

First of all, Rust corner rules: 
1. single owner, single place of responsibility
2. xxx

## Ownership 
```Rust
fn myprint(v: Vec<i32>){
        printl!("{:?}", v);
}

fn main(){
        let v = vec![1,8,4,7];
        myprint(v); // ownership is transferred to function myprint
        myprint(v); // compile-time error, cause 
}
```
Because resource is moved from the old location (say, a local variable) to the new location (a function argument), this process of transferring is also called **moving**.
This is also why when you try to accessed a data structure of variable whose ownership has been transferred elsewhere, you will get `error[E0382]: use of moved value`, e.g., 
<img class='center' src="{{site.baseurl}}/images/2020/error.e0382.png" width="650">

## Borrowing
Borrowing is like reference in C, it allows us to have multiple references to a resource.

## Reference 
1. [Understanding Rust: ownership, borrowing, lifetimes](https://medium.com/@bugaevc/understanding-rust-ownership-borrowing-lifetimes-ff9ee9f79a9c)
2. [Rust Means Never Having to Close a Socket](https://blog.skylight.io/rust-means-never-having-to-close-a-socket/)
3. [Rust doc](https://doc.rust-lang.org/1.8.0/book/ownership.html)










