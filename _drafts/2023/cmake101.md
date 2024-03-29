---
layout:     post
title:      Cmake 101
date:       2023-02-12
tags:
categories: 
- C++
---


```bash
cmake_minimum_required(VERSION 3.10)
project(MyProject)

add_executable(main main.cpp)
```

手动编写 `makefile` 对于新手而言比较痛苦，对于专业人士而言也比较浪费时间。作为一个跨平台的构建工具，CMake 可以根据用户的需求生成各种各样的 Makefile 或者 Project 文件。CMake 的优点如下：
- 跨平台性：CMake 可以生成不同平台下的工程文件，因此可以轻松地在多个平台上编译和构建项目。
- 易于使用：CMake 提供了一个简单的语法，可以简化构建过程。
- 易于维护：CMake 可以简化项目的配置，因此可以方便地维护项目。
- 支持各种构建工具：CMake 可以生成不同的构建工具的工程文件，如 Unix Makefiles、Microsoft Visual Studio 项目、Xcode 项
- 可扩展性：CMake 支持自定义命令和模块，因此可以方便地扩展其功能。

# Q&A
Q: 如何指定 C++ 版本？

A: 
```bash
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```
再比如 ，C++ 编译时常用的参数有：
- `-std=c++xx`：指定 C++ 版本，其中 `xx` 表示版本号，如 `-std=c++14`。
- `-O1, -O2, -O3`：指定优化级别，其中 `-O1` 表示低级优化，`-O2` 表示中级优化，`-O3` 表示高级优化。
- `-Wall`：启用所有警告，帮助捕捉代码中的问题。
- `-Werror`：将警告视为错误，防止问题代码编译通过。
- `-g`：生成调试信息，方便使用调试器调试代码。
- `-static`：指定静态链接，生成的可执行文件不依赖于共享库。
- `-march=native`：指定编译时优化为当前计算机的架构，生成的代码更快。

在 `CMakeLists.txt` 中使用 `add_compile_options` 命令添加以上编译选项:

```bash
add_compile_options(-std=c++14 -O2 -Wall -Werror -g)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
```


# 文档
- [CMake 文档翻译](https://www.jianshu.com/p/7326f9167fae)
