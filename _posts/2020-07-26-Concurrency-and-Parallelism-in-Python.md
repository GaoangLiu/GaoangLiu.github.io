---
layout:     post
title:      Concurrency and Parallelism in Python
date:       2020-07-26
tags: [python, concurrency]
categories: 
- python
---

# GIL 
GIL(Global Interpreter Lock) 全局解释锁，保护 Python 内部对象的全局锁，保障解释器的线程安全。
1. 实现是内核级互斥(GIL保护的是解释器级别的数据)，不保证用户级互斥。即在不加锁情况下，由用户引入的数据仍然可以被同时访问、修改(从而可能造成数据丢失)
2. 并非 Python 的特性，而是解释器引入的概念。JPython, IronPython 中没有 GIL，CPython，PyPy 中存在
3. 本质上是互斥锁， 将并发运行变成串行


# 并发/并行
并发，多个事件在同一时间间隔内运行，即一次做多件事，事件之间穿插运行，但不能同时运行。

并行，多个事件在同一时间发生，严格意义上的同时发生。并行任务需要多 CPU 支持

直观上，并发看起来像计算机在同一时间做着许多不同的事情，但这实际上是一种假象。而并行是计算机确实在同一时间内做许多不同的事情。

## Python 多线程

> "Python 线程毫无作用。" -- David Beazly ([Generators: The Final Frontier](http://www.dabeaz.com/finalgenerator/))

在带有 GIL 的 Python 解释器下，一次只能有一个线程执行 Python 字节码，因此多线程程序并不能发挥多核 CPU 的优势。比如，一个使用了多线程的计算密集型程序仍然只在一个 CPU 上运行。
多线程下线程执行方式
1. 获取 GIL
2. 运行直到 sleep 或者被挂起
3. 释放 GIL

多线程并不一无是处，标准库中阻塞型 I/O 函数都会在等待操作系统返回结果时释放 GIL(`time.sleep`函数也会释放 GIL)，因此 I/O 密集型任务(文件处理、网络爬虫等)可从多线程中受益。


## `concurrent.futures` 模块
[`concurrent.futures` 模块](https://docs.python.org/3/library/concurrent.futures.html) 提供了一个高层面的接口来实现异步调用，它提供了两个类 `ProcessPoolExecutor` 及 `ThreadPoolExecutor` 且实现了通用的 `Executor` 接口。 示例

```python
with futures.ThreadPoolExecutor(max_workers=10) as e:
    e.submit(func, **args)
```
对于 CPU 密集型任务而言，能过 `ProcessPoolExecutor` 可以绕开 GIL，利用所有可用的 CPU 核心。在不指定 `max_workers` 参数时，程序会使用 `os.cpu_count()` 个 CPU。

`Executor` 可通过 `.submit()` 及 `.map()` 来提交任务，二者区别在于: 
1. `submit` 提交单个回调任务; `map` 可提交一批任务，以异步的形式将函数作用于可迭代对象上
1. `submit` 可处理不同的可调用对象与参数; 而`map`只能处理参数不同的同一可调用对象
2. `submit` 返回一个`Future`对象，可通过 `done` 方法判断任务是否结束，通过 `future.result()` 来获取结果; `map`直接返回由结果组成的生成器

## 进程与线程
二者都可以实现多任务

| 进程 | 线程 | 
|:--- |:---- |
| 资源分配的基本单位 | CPU 调度的基本单位 |
| 不共享全局变量，创建子进程时对主进程进行资源拷贝 | 线程之间共享进程的所有资源 |
| 创建进程需要 copy 资源，开销较大 | 基本上不拥有系统资源，创建、调度开销很小 |



