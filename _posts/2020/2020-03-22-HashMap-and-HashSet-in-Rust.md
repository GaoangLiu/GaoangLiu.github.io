---
layout: post
title: HashMap and HashSet in Rust
date: 2020-03-22
tags: rust hashset
categories: 
author: berrysleaf
---
* content
{:toc}


# HashMap
While a vector stores values by index, `HashMap` stores value by key, which can be **booleans, integers, strings**, or any other type that implements the `Eq` and `Hash` traits.




Like vectors, hash maps are **homogeneous**: all of the keys must have the same type, and all of the values must have the same type.


## Constructing a HashMap
We can create a `HashMap` with `new` and add elements with `insert`. E.g., 
```rust
let mut hm = HashMap::new(); 
hm.insert(1, 10);
hm.insert(2, 20);
```

Another way of constructing a hash map is by using the `collect` method on a vector of tuples. E.g., 
```rust
let scores = vec![90, 80];
let judges = vec![String::from("Excellent"), String::from("Good")];
let hm: HashMap<_, _> = scores.iter().zip(judges.iter()).collect(); 
```

## Iterate 
```rust
for (k, v) in &hm {
        println!("{}: {}", k, );
}
```

## Update HashMap 
Use `map.insert(k, v)` to simply overwriting the value of key `k`, or if we only want to insert value for a key that does not exist, we can use `entry`:
```rust
map.entry(k).or_insert(v);
```

This `or_insert` will return the value for the corresponding key if that key exists, and if not, inserts the parameter as the new value for this key and returns the new value.

## Update a value based on old value 
For instance, we want to increase the value of a key if it exists in hash map, otherwise set the value to be 1. 
```rust
let text = "hello world wonderful world";
let mut map = HashMap::new();
for word in text.split_whitespace() {
    let count = map.entry(word).or_insert(0);
    *count += 1; 
    // or we can merge above two lines into one: *map.entry(word).or_insert(0);
}
```

# HashSet

A hash set is a `HashMap` where the value is (), i.e., it is a wrapper around `HashMap<T, ()>)`.

## Usage
### Examples
```rust
use std::collections::HashMap; 
let mut users = HashSet::new(); 
users.insert("John Smith".to_string());
users.insert("Jason Marz".to_string());
users.insert("Carmen Sandiago".to_string());
```

## Build HashMap from a vector
There are differences between the following two ways of constructing `HashMap` from a `Vector`:

```rust
use std::iter::FromIterator;
let hs: HashSet<i32> = HashSet::from_iter(arr); // this will move ownership to hs, or 
let hs: HashSet<i32> = HashSet::from_iter(arr.iter().cloned()); // this will preserve the ownership of arr 
```
