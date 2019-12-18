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
typedef unsigned long size_t;
``` 

size_t在32位架构上是4字节，在64位架构上是8字节。

一般来说，在表示数据结构大小的时候，尽量使用 `size_t` 。原因，1. 代码表述清晰，一眼就可以看出这是在表示一个对象的长度 ； 2. 表示范围比 `int` 要大，特别是表示 vector 或者其他 container 的长度时 `size_t` 可以确保不出出现溢出等问题。

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
std::bitset<8> b3(bit_string, 0, 1);   // [0, 0, 0, 0, 0, 0, 0, 1] 
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

**注：** 算法中 `bitset<10001>` 大小为 10001 是因为原问题中集合大小限制在 200，元数大小限制在 100，因此我们的 target 不太于 10000 。
