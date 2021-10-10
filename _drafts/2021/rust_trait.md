---
layout:     post
title:      Rust Trait
date:       2021-10-10
tags: [trait]
categories: 
- rust
---

A **trait**, called **interfaces** in other languages, tells the Rust compiler about functionality a particular type has and can share with other types. We can use traits to define shared behavior in an abstract way. We can use trait bounds to specify that a generic type can be any type that has certain behavior.

## Defining a Trait

The behavior of a type consists of the methods that we can call on that type. Different types share the same behavior if we can call the same methods on all of them. A trait definition is a way of grouping method signatures to define a set of behaviors needed to accomplish some purpose.

We can define a trait, as shown in the following example, providing a method to display summaries of data, whether the data is of type `Post` or `Article`.  


```rust
pub trait Summary{
    fn summarize(&self)->String;
}
```

### Semantic
A trait can have multiple methods in its body: the method signatures are listed one per line and each line ends in a semicolon.


## Implementating a Trait on a Type
Implementing a trait on a type is similar to implementing regular methods. The difference is that after `impl`, we put the trait name that we want to implement, then use the `for` keyword, and then specify the name of the type we want to implement the trait for. 


```rust
pub struct Post{
    pub title: String,
    pub author: String,
    pub content: String,
}

impl Summary for Post{
    fn summarize(&self) -> String {
        format!("{}, by {} ({})", self.title, self.author, self.content)
    }
}

pub struct Article{
    pub title: String,
    pub author: String,
    pub introduction: String,
}

impl Summary for Article{
    fn summarize(&self)-> String{
        format!("{}, {}, {}", self.title, self.author, self.introduction)
    }
}

```


## Default Implementations
Sometimes it’s useful to have default behavior for some or all of the methods in a trait instead of requiring implementations for all methods on every type. Then, as we implement the trait on a particular type, we can keep or override each method’s default behavior.

```rust
pub trait Summary {
    fn summarize(&self) -> String {
        String::from("SOME SUMMARY.")
    }
}
```

Default implementations can call other methods in the same trait, even if those other methods don’t have a default implementation. In this way, a trait can provide a lot of useful functionality and only require implementors to specify a small part of it. For example, we could define the Summary trait to have a summarize_author method whose implementation is required, and then define a summarize method that has a default implementation that calls the summarize_author method:

```rust
pub trait Summary {
    fn summarize_author(&self) -> String;

    fn summarize(&self) -> String {
        format!("(Read more from {}...)", self.summarize_author())
    }
}

impl Summary for Post {
    fn summarize_author(&self) -> String {
        format!("@{}", self.author)
    }
}
```


# References
- [rust doc on Traits](https://doc.rust-lang.org/book/ch10-02-traits.html)

