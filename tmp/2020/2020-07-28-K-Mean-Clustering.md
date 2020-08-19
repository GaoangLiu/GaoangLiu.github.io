---
layout: post
title: K-Mean Clustering
date: 2020-07-28
tags: machine_learning k-mean
categories: machine_learning
author: GaoangLau
---
* content
{:toc}


# 线性同余法
1. 确定初始种子 $$n_0$$
2. 迭代(递归) $$n_{i+1} = (a \cdot n_i + b) \% m$$



则可产生 $$[0, m-1]$$ 之间的随机数，这里考虑的 $$a, n, b, m$$ 均为非负整数。

方法简单高效，但产生的随机数是周期性的，且周期 $$\leq m$$。证明
1. 存在周期性。反证，假设对 $$\forall (i, j) n_i \neq (a \cdot n_j + b) \% m $$，那么在进行 $$m$$ 次计算后共生成了 $$m$$ 个互不相同的随机数，记最后一个随机数为 $$n_{m-1}$$，显然有 $$ n_m = (a \cdot n_{n-1} + b)  \% m$$ 为一个新的随机数，且不在 $$[0, m-1]$$ 之间，impossible
2. 周期 $$\leq m$$
    1. 取 $$a=1$$, 则周期为 $$m$$
    2. 取 $$a = 2, b = 0$$, 取$$m$$为任意大于 2 的偶数，可见一定无法生成奇数随机数，故周期必然小于 $$m$$

通过特定条件的参数选择 $$a, b, m$$，我们可以构造一个周期刚好为$$m$$的序列。在这个序列中，每个数出现的频率是相同的，也即服从平均分布。

# Mersenne Twister 
基于有限二进制字段上的矩阵线性递归 ... 

```python
def _int32(x):
    return int(0xFFFFFFFF & x)

class MT19937:
    def __init__(self, seed):
        self.mt = [0] * 624
        self.mt[0] = seed
        self.mti = 0
        for i in range(1, 624):
            self.mt[i] = _int32(1812433253 * (self.mt[i - 1] ^ self.mt[i - 1] >> 30) + i)


    def extract_number(self):
        if self.mti == 0:
            self.twist()
        y = self.mt[self.mti]
        y = y ^ y >> 11
        y = y ^ y << 7 & 2636928640
        y = y ^ y << 15 & 4022730752
        y = y ^ y >> 18
        self.mti = (self.mti + 1) % 624
        return _int32(y)


    def twist(self):
        for i in range(0, 624):
            y = _int32((self.mt[i] & 0x80000000) + (self.mt[(i + 1) % 624] & 0x7fffffff))
            self.mt[i] = (y >> 1) ^ self.mt[(i + 397) % 624]

            if y % 2 != 0:
                self.mt[i] = self.mt[i] ^ 0x9908b0df

print(MT19937(42).extract_number())
```

特点：
1. 周期为 $$2^{19937}-1$$，质量相对较高。长周期不保证高质量
2. 在 $$1 \leq k \leq 623$$的维度之间都可以均等分布

### 梅森素数
形如 $$M_n = 2^n - 1$$ 的数为梅森数，如果它还是素数，则称为格林素数。截至到今日(2020-07)，只找到51位梅森素数，当前最大的梅森素数为 $$M_{82589933}$$

上面 MT 算法的周期即为一个梅森素数。