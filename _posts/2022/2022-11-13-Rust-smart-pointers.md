---
layout: post
title: Rust smart pointers
date: 2022-11-13
tags: rust
categories: rust
author: gaoangliu
---
* content
{:toc}


又是捞鱼的一天，这一天 C++ 程序员与 Rust 程序员再一次在某知名交友网站对上了线。Rust 程序员吐槽 C++ 的包管理混乱、内存安全及螺旋上升的精通难度等问题。C++ 程序员不予置评，而是扶了扶厚厚的镜片，轻微整理下头顶上方愈加稀疏的头发，对 Rust 程序员微微一笑，说：“写个链表试试”。




<img src="https://file.ddot.cc/imagehost/2022/9afd0dfc-d422-4646-bd1e-090c4a3ee438.png" width=678pt>

使用 C++ 写一个链表简直小菜一碟，在 Rust 程序员还在 `cargo new my_awesome_linkedlist_project` 的时候，C++ 程序员已经把节点结构写完了。

```cpp
struct Node {
    int value;
    Node* next;
};
```

Rust 程序员看了看 C++ 程序员的代码，然后说：“你这个写法是不是可能内存泄漏啊”？

C++ 程序员说：“只要代码中不使用就不会有内存泄露，但如果一定要用的话，这个问题也好解决，我用  `std::unique_ptr` 强化一下就 okay 了” 。 


```cpp
template <typename T>
struct Node {
    T value;
    std::unique_ptr<Node<T>> next;
};
std::unique_ptr<Node<int>> head;
```

这就过分了，他不仅快速写了一个链表，还顺便秀了一下模板。Rust 程序员不甘示弱，说：“我用 `Box` 就能写出来啊，即快捷又安全”。

```rust
struct Node {
    value: i32,
    next: Option<Box<Node>>,
}
let mut head = Box::new(Node { value: 0, next: None });
```

对 C++ 程序员来说，这个 `Box` 非常陌生。 `Box<T>` 是 Rust 中最常用的智能指针（[source](https://course.rs/advance/smart-pointer/box.html)），作用是将一个值放在堆上，而不是栈上。这样一来，当 `Box<T>` 离开作用域时，它所指向的值也会被销毁。这样一来，就不会出现内存泄漏的问题。

这里使用 `Box<T>` 的另一个原因是，Rust 在编译时需要知道 `Node` 的大小，下面的代码是一个递归结构写法， Rust 编译器无法计算出 `Node` 的大小，因此无法编译通过。但 `Box<T>` 是一个指针，它的空间大小我们是知道的[note-1]，且指针的大小并不会根据其指向的数据量而改变。

> note-1: 在 64 位系统下，一个指针的大小是 8 字节，指针类型是 `usize`。 - [stackoverflow](https://stackoverflow.com/questions/42358389/why-is-the-size-of-a-pointer-to-something-on-the-heap-larger-than-the-size-of-a)

```rust
struct Node {
    value: i32,
    next: Option<Node>,
}
```


# `Box<T>` 
## 场景 
由于 Box 是简单的封装，除了将值存储在堆上外，并没有其它性能上的损耗。而性能和功能往往是鱼和熊掌，因此 Box 相比其它智能指针，功能较为单一，可以在以下场景中使用它：

1. 特意的将数据分配在堆上
2. 数据较大时，又不想在转移所有权时进行数据拷贝
3. 类型的大小在编译期无法确定，但是我们又需要固定大小的类型时
4. 特征对象，用于说明对象实现了一个特征(trait)，而不是某个特定的类型

### 场景 1: 特意的将数据分配在堆上
```rust
let b = Box::new(5);
println!("b = {}", b);
```

### 场景 3: 类型的大小在编译期无法确定
Rust 需要在编译时知道类型占用多少空间，如果一种类型在编译时无法知道具体的大小，那么被称为动态大小类型 DST。

递归类型（recursive type）是 DST 的一种，其值的一部分是相同类型的另一个值。这种值的嵌套理论上可以无限的进行下去，所以在 Rust 中直接定义这种类型是不允许的。但是，通过 `Box<T>` 将递归类型放在堆上，就可以避免这个问题。

```rust
enum Recursive {
    Value(i32),
    Link(Box<Recursive>),
}
```

### 场景 2: 数据较大时，又不想在转移所有权时进行数据拷贝
当栈上数据转移所有权时，实际上是把数据拷贝了一份，最终新旧变量各自拥有不同的数据，因此所有权并未转移。

而堆上则不然，底层数据并不会被拷贝，转移所有权仅仅是复制一份栈中的指针，再将新的指针赋予新的变量，然后让拥有旧指针的变量失效，最终完成了所有权的转移：

```rust
fn main() {
    let a = [0; 100000];
    let b = a;

    // a 和 b 都拥有各自的栈上数组，因此不会报错
    println!("{:?}", a.len());
    println!("{:?}", b.len());


    let a = Box::new([0;1000]);
    // 将堆上数组的所有权转移给 arr1，由于数据在堆上，因此仅仅拷贝了智能指针的结构体，底层数据并没有被拷贝
    // 所有权顺利转移给 arr1，arr 不再拥有所有权
    let b = arr;
    // println!("{:?}", b.len());
    // 由于 arr 不再拥有底层数组的所有权，因此下面代码将报错
    // println!("{:?}", a.len());
}
```



# Rc<T>
Rust 所有权机制要求一个值只能有一个所有者，这大多数场景下都没有问题，但是有时候我们需要多个所有者，比如多个线程同时访问一个变量，或者多个函数同时访问一个变量。为了解决此类问题，Rust 在所有权机制之外又引入了额外的措施来简化相应的实现：通过**引用计数**的方式，允许一个数据资源在同一时刻拥有多个所有者。

这种实现机制就是 Rc 和 Arc，前者适用于单线程，后者适用于多线程。

Rc 是 reference counting 的缩写，记录是一个变量的引用计数。 通过记录一个数据被引用的次数来确定该数据是否正在被使用。当引用次数归零时，就代表该数据不再被使用，就可以被清理释放。

使用场景：当我们希望在堆上分配一个对象供程序的多个部分使用且无法确定哪个部分最后一个结束时，就可以使用 Rc 成为数据值的所有者。

举个例子：
```rust
use std::rc::Rc;
fn main(){
    let a = Rc::new(String::from("hello"));
    let b = Rc::clone(&a);

    println!("a = {}, b = {}", a, b);
    println!("strong count = {}, weak count = {}", Rc::strong_count(&a), Rc::weak_count(&a));
}
```

以上代码我们使用 `Rc::new` 创建了一个新的 `Rc<String>` 智能指针并赋给变量 `a`，该指针指向底层的字符串数据。

智能指针 `Rc<T>` 在创建时，还会将引用计数加 1，此时获取引用计数的关联函数 `Rc::strong_count` 返回的值将是 1。

使用 `Rc::clone` 克隆了一份智能指针 `Rc<String>`，并将该智能指针的引用计数增加到 2。这里的 `clone` 其实是浅拷贝，只复制了智能指针并增加了引用计数，并没有克隆底层数据。当然，这里也可以使用 `b.clone()`，这个地方与 `Rc::clone` 没有本质区别，仅复制了智能指针，但 `Rc::clone` 更清晰。关于 `Rc::clone(&rc)` 与 `rc.clone()` 的差异，可参考 [Is there any difference between Rc::clone(&rc) and rc.clone() in Rust? Is there any compilation optimizations happen based on that?](https://stackoverflow.com/questions/61949769/is-there-any-difference-between-rcclonerc-and-rc-clone-in-rust-is-there)。 


# Arc<T>
`Rc<T>` 没有实现 `Send` 特征，因此不能在线程间安全的传递。更深层的原因：由于 `Rc<T>` 需要管理引用计数，但是该计数器并没有使用任何并发原语，因此无法实现原子化的计数操作，最终会导致计数错误。

这时候 `Arc<T>` 就闪亮登场了。 `Arc` 是 `Atomic Rc` 的缩写，顾名思义：原子化的 `Rc<T>` 智能指针。

一个例子：
```rust
use std::sync::Arc;
use std::thread;

fn main() {
    let s = Arc::new(String::from("多线程漫游者"));
    for _ in 0..10 {
        let s = Arc::clone(&s);
        let handle = thread::spawn(move || {
           println!("{}", s)
        });
    }
}
```

注：Arc 定义在 `std::sync::Arc` 模块下。 

总结而言，Rc 和 Arc 的区别在于，后者是原子化实现的引用计数，因此是线程安全的，可以用于多线程中共享数据。但同时，这两者都是只读的，如果想要实现内部数据可修改，必须配合内部可变性 RefCell 或者互斥锁 Mutex 来一起使用。


# 参考 
- [Rust 中的智能指针](https://kaisery.github.io/trpl-zh-cn/ch15-00-smart-pointers.html)
- [Rust语言圣经(Rust Course)](https://course.rs/advance/smart-pointer/box.html)

