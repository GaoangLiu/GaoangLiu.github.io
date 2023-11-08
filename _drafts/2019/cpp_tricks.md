---
layout:     post
title:      Data Structures and Tricks in C++
date:       2019-11-27
tags: [cpp, data structure]
categories: 
- cpp
---

## 整形
- `short` 至少 16 位
- `int` 至少和 short 一样长
- `long` 至少 32 位，且至少和 `int` 一样长
- `long long` 至少 64 位，且至少和 `long` 一样长

`int` 并不总是 32 位，在老式 IBM PC 的实现中，`int` 的宽度为 16 位，而在 Windows XP、Windows Vista、Windows 7 及其他多个微型计算机中，为 32 位 (其中 long 也为 32 位)

## `long` v.s. `int64_t`
`int64_t` 一定可以保证是 64 位，但 `long` 未必总是是 64 位。 

另外关于在代码中使用如 `int16_t, int32_t, int64_t` 这样精准、清晰的表示方式，还是简单的 `int, long` ，没有较为统一的意见。

个人观点：simple is better 。 如果没有充足的理由非得使用刚好 32 或者 62 位的变量，还是建议直接使用较为通用的声明方式。

支持 `int64_t` 类型的编译器必然支持 `long long`，但 pre-C++ 2011 的编译器不支持 `int64_t`。

## `size_t` v.s. `int` 
`size_t` 的真实类型与操作系统有关，其取值范围是目标平台下最大可能的数组尺寸。一些平台下`size_t`的范围小于`int`的正数范围,又或者大于 `unsigned int`。 使用`int`既有可能浪费，又有可能范围不够大。


在32位架构中被普遍定义为：
```cpp
typedef unsigned int size_t;
```
而在64位架构中被定义为：
```cpp
typedef unsigned long size_t;
```

`size_t` 在32位架构上是4字节，在64位架构上是8字节。而 `int` 在 32/64 位系统下都是 4 字节。

一般来说，在表示数据结构大小的时候，尽量使用 `size_t` 。原因:
1. 代码表述清晰，一眼就可以看出这是在表示一个对象的长度 ； 
2. 表示范围比 `int` 要大，特别是表示 vector 或者其他 container 的长度时 `size_t` 可以确保不出出现溢出等问题。

### 陷阱

示例 1: `size_t` 为无符号类型，做减法时确保不要出现负值。
```cpp
string s = "abc";
int n = 2;
size_t m = 2; 
cout << n - s.size() << endl;
```
这里使用`int`而不是`size_t`来声明 `n`，在我们 64 位 Mac OS 系统中的输出是：18446744073709551615 (2**64-1)，这是因为 `s.size()` 返回值是 `size_type` 一个无符号整数，而编译器在`int`与`size_type`做减法时，都视为了无符号整数。

<img src="http://git.io/JJ9R9" width="500px" alt="size_t example">


<!--- 示例 2: 在使用 `str.find(some_substr)` 时 `str::npos` 是 `size_t` 类型
```cpp
string s = "A very long string ..."; 
int idx = s.find("MISS");
if (idx == str::npos) {
    // subroutine
}
```
 --->


## 随机 
生成随机浮点数
```cpp
random_device rd; 
mt19937 gen(rd());
uniform_real_distribution<double>dis(lower_bound, upper_bound);
double random_double = dis(gen);
```



## 字符串
### 字符转化为字符串的 9 种方式
1. `char c = 'a'; string s (1,c);`，类构造器 `string(size_t n, char c)`
2. `char c = 'a'; string s; stringstream ss; ss << c; ss >> s; `
3. `string s; s.push_back(c); ` ，使用 `push_back` 方法 
4. `s += c;` 
5. `string s; s = c; `
6. `s.append(1, c);`, `append(size_t n, char c)` 添加 `n` 个字符 
7. `s.assign(1, c);`
8. `s.insert(0, 1, c);` ，`insert(site_t pos, site_t n, char c)` 在位置 `pos` 插入 `n` 个字符 
9. `s.replace(0, 1, 1, c);`，`replace(size_t pos, site_t len, site_t n, char c)` 从位置 `pos` 开始，用 `n` 个字符 `c` 替换长度为 `len` 的部分

### `std::string_view`
从 C++17 开始新加入的类，它提供一个字符串的视图，即可以通过这个类以各种方法"观测"字符串，但无法修改字符串。它的特点：

1. 通过调用 `string_view` 构造器可将字符串转换为 `string_view` 对象。
2. `string` 可隐式转换为 `string_view`。
3. `string_view` 对所指向的字符串没有所有权。可用来取代 `const char*` 和 `const string&`，以避免不必要的内存分配。
4. `string_view` 的成员函数即对外接口与 `string` 相类似，但只包含读取字符串内容的部分。
5. `string_view::substr()` 的返回值类型是 `string_view`，不产生新的字符串，不会进行内存分配。其空间复杂度为常数

### [`std::lexicographical_compare`](https://en.cppreference.com/w/cpp/algorithm/lexicographical_compare)
STL 泛型算法函数来比较两个序列大小 

用法示例：
```cpp
td::string str = "aaaaaaaaaaaaaaaaaaaaa realllllllllllllllly long string";

//Bad way - 'string::substr' returns a new string (expensive if the string is long)
std::cout << str.substr(15, 10) << '\n';

//Good way - No copies are created!
std::string_view view = str;

// string_view::substr returns a new string_view
std::cout << view.substr(15, 10) << '\n';
```

`string_view` 的 `substr` 方法返回的是字符串的“视图”，为常数复杂度


# 数据结构
## bitset
区别于 vector, array，bitset 索引**从右向左**。

初始化方法 
```cpp
string s = "100011";
bitset<8> ba(s);  // generate [0, 0, 1, 0, 0, 0, 1, 1]

// string from position 2 till end 
std::bitset<8> b2(bit_string, 2);      // [0, 0, 0, 0, 0, 0, 1, 1] 

// string from position 2 till next 3 positions 
std::bitset<8> b3(bit_string, 2, 3);   // [0, 0, 0, 0, 0, 0, 0, 1] 
```

函数 
1. `count()`, 返回 ture 的个数 
2. `size()`, 大小
3. `any() | all()`, 存在或者所有单位为 true
4. `set()`, 重置所有单元为 true
5. `set(3)`, 置第 4 个单元为 true
6. `set(3, 0)`, 置第 4 个单元为 false 
7. `reset()`, 重置所有单元为 false
8. `flip()`, 反转单元值

### 示例
[子集合划分](https://leetcode.com/problems/partition-equal-subset-sum/description/): 给定一个集合 $$S$$ ，是否可以将其分两个子集合使得其和相等？更一般的，给定一个整数 $$N$$，能否找到一个子集体 $$T \subseteq \text{s.t.,} \sum T = N $$ ?

bitset 解法：
```cpp
bool can_partition(vector<int> nums){
    int sum = 0; 
    bitset<10001> bits(1);
    for(auto n: nums){
        sum += n; 
        bits |= bits << n;
    }
    return (sum % 2 == 1) && bits[sum / 2];
}
```

普通的 DP(dynamic programming) 解法 
```cpp
bool can_partition(vector<int> nums){
    int sum = accumulate(nums.begin(), nums.end(), 0);
    if (sum % 2 == 1) return false; 
    vector<bool> dp(sum / 2 + 1, false);
    dp[0] = true; 
    for (auto n: nums) {
        vector<bool> nxtdp(sum / 2 + 1, false);
        for (size_t j = 0; j < dp.size(); j ++){
            if (dp[j]) nxtdp[j + n] = true, nxtd[j] = true; 
        }
        dp = nxtdp; 
    }
    return dp[sum / 2];
}
```
其中 `dp[i]` 表示能否从集合中得到一个子集合，使得其和为 `i`。

算法的思想是，假设 `dp` 为前 m 个元素可能构成的子集合再取和的所有情况，考虑第 m+1 个元素 $$S[m]$$，如果 dp[287] = true (即$$\exists T \subseteq S[:m] \text{s.t., } \sum T = 287$$)，那么 dp[287+$$S[m]$$] = true 。 

bitset 的解法本质上是利用 **bitset 代替 DP(dynamic programming) 表格，用位运算取代 DP 的迭代过程**。 
bits[i] 与 dp[i] 表示的意义一致。 

Tiny example, S = [2, 3, 4], bits 初始为 1 ， 迭代如下: 
1. n = 2, bits = 101 ， 表示集合 $$\{2\}$$ 可构成 0, 2
2. n = 3, bits = 101101 ， 表示集合 $$\{2, 3\}$$ 可构成 0, 2, 3, 5
3. n = 4, bits = 1011111101 ， 表示集合 $$\{2, 3, 4\}$$ 可构成 0, 2, 3, 4, 5, 6, 7, 9

**注：** 算法中 `bitset<10001>` 大小为 10001 是因为原问题中集合大小限制在 200，元素大小限制在 100，因此我们的 target 不太于 10000 。


## Multiset 
是 `Set` 数据结构的变体，具备以下特性：
1. 按大小顺序存储数值 (如元素顺序不重要，可能考虑使用 `unordered_multiset`)
2. 元素可以不唯一
3. 只能插入、删除元素，无法修改
4. 通过迭代器可删除多个元素
5. 通过迭代器对元素进行迭代
6. 以 BST 的形式实现

举例：
```cpp
multiset <int, greater<int>> ms; 
ms.insert(40);  // 40
ms.insert(50);  // 50, 40
ms.insert(60);  // 60, 50, 40
ms.insert(50);  // 60, 50, 50, 40
```

支持的方法与 `set` 相同，包括 `size(), empty(), erase(), clear(), count() ` ...


<!-- TODO -->
# Standard Template Library (STL)

## Map 
`map` v.s., `unordered_map`

|           |    `map`      |  `unordered_map` |
|----------|:-------------:|:------:|
| 元素存在顺序 |  YES | NO |
| 实现 |  自平衡BST   |  哈希表 |
| 查找时间 | $$\text{log}(n)$$ | 平均 $$O(1)$$ / 最坏 $$O(n)$$ |
| 插入时间 | $$\text{log}(n)$$ + 平衡时间 | 平均 $$O(1)$$ / 最坏 $$O(n)$$ |
| 删除时间 | $$\text{log}(n)$$ + 平衡时间 | 平均 $$O(1)$$ / 最坏 $$O(n)$$ |

## std::vector v.s. std::array

在 C++11 中，STL 中提拱了一个新的容器 `std::array`，该容器取代了 C 类型的数组>

`array` 与 `vector` 的相同点在于：
1. 都对下标运算符 `[]` 进行了重载，可能使用标准数组表示法访问各个元素
2. 都实现了 `front()`, `back()`, `size()`, `empty()` 等方法，提供迭代器(`begin()`, `end()`)遍历机制


区别:
* `array` 使用栈(静态内存分配)，编译时确定大小、更轻量、更效率，而 `vector` 使用堆(动态存储)，使用 `new`, `delete` 来管理内存; 
* `vector` 属于可变长容器，可以动态更改容器容量； `array` 属于定长容量，初始化必须指定大小。
    * 从语法上讲，`vector` 提供但 `array` 没有的 `resize()`, `erase()` 等方法归结于其容量是否可变动 


### array vector 实现方法对比

| Function | array | vector |
|:---------|:-------|:-------|
|`constructor`, `destructor`  |  ❌ | ✔ |
|`push_back()`, `pop_back()` |  ❌ | ✔ |
|`resize()`, `capacity()`, `reserve()` |  ❌ | ✔ |
|`erase()`, `clear()` |  ❌ | ✔ |
|`empty()`, `size()`, `max_size()`  |  ✔ | ✔ |
|`at()`, `front()`, `back()`  |  ✔ | ✔ |
|`assign()`, `swap()`  |  ✔ | ✔ |
|`operator = \< == []`  |  ✔ | ✔ |


### 小结
* `array` 有的特性，`vector` 基本上都有，反之不然。二者最大的区别在于容量是否可以动态变化，方法的差异也体现在这一点上； 
* `vector` 方便安全，通常情况下都应该作为首选项，特别是在元素个数会动态增长/减少或者提前不可知的情况下。而如果预先确定元素个数且个数不多，可以考虑使用 `std::array`。



