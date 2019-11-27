---
layout:     post
title:      Data Structures and Tricks in C++
date:       2019-11-27
tags: [cpp, data structure]
categories: 
- cpp
---

## `size_t` vs `int` 
`size_t` 的真实类型与操作系统有关，其取值范围是目标平台下最大可能的数组尺寸。

在32位架构中被普遍定义为：
```cpp
typedef unsigned int size_t;
```
而在64位架构中被定义为：
```cpp
typedef unsigned long size_t;``` 

size_t在32位架构上是4字节，在64位架构上是8字节。

一般来说，在表示数据结构大小的时候，尽量使用 `size_t` 。原因，1. 代码表述清晰，一眼就可以看出这是在表示一个对象的长度 ； 2. 表示范围比 `int` 要大，特别是表示 vector 或者其他 container 的长度时 `size_t` 可以确保不出出现溢出等问题。

