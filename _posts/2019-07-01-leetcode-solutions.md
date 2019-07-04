---
layout:     post
title:      Leetcode solutions
subtitle:   
date:       2019-07-03
author:     ssrzz
catalog: 	true
tags:
  - algorithm
  - python
---

## 313 super ugly numbers

[Que](https://leetcode.com/problems/super-ugly-number/):  Super ugly numbers (SUN hereafter) are positive numbers whose all prime factors are in the  given prime list primes of size $$k$$,  (e.g., primes = [2, 7, 13, 19])  Write a program to find the $$n$$-th super ugly number.

### Method 1: Brute Force

Idea: searching from 1 to some $$m>>n$$ to find the $$n$$-th number which is super ugly. 

Comment: bad solution when $$n$$ is large, as we have to decompose each number $$1 < i < m$$ to figure out whether it is super ugly. 

### Method 2: Generating SUNs incrementally

Idea:  First SUN is surely $$1$$, the second SUN is $$\text{min}(primes) = 2$$ (assuming $$primes=[2,7,13,19]$$) , the third ? We have to do some calculations, let $$U = [1, 2]$$ be the first two generated SUNs, $$C = [p * u  \text{ for } p \in primes \text{  for  } u \in U ] \backslash U$$. Then $$U_3 = \text{min}(C) = 4$$ , making $$U=[1,2,4]$$. 

By repeating the previous procedure, we can find the $$n$$-th SUN within time $$O(n^2 * m)$$, where $$m$$ is the size of $$primes$$. 

Or, we can speed up the procedure by keeping track of numbers we've calculated before using an auxiliary array $$idx$$ , $$idx[i]$$ means previous  $$(idx[i]-1) $$ SUNs with one prime factor being $$primes[i]$$ have been calcuated, with the results stored either in $$U$$, or in the candidiate set $$C$$.

More specifically, initialize $$idx = [0,0,0,0] , U = [1], C = [p * u  \text{ for } p \in primes \text{  for  } u \in U ] \backslash U = primes $$ , the second SUN is $$\text{min} (C) = 1 * 2 $$, we update $$idx = [1(0+1),0,0,0]$$

3-rd SUN is $$min(C')$$, where $$C'  = [U_{idx[0]} * primes[0], U_{idx[1]} * primes[1], …] = [4,7,13,19]$$. At this time, the lucky $$primes[i]$$ remains $$2$$ with $$i == 0$$, so we update $$idx = [2(1+1),0,0,0]$$. 

4-th: $$C'' = [8,7,13,19], U = [1,2,4,7], idx = [2,1,0,0]$$

5-th: $$C'' = [8,14,13,19], U = [1,2,4,7,8], idx = [3,1,0,0]$$

6-th: $$C'' = [14,14,13,19], U = [1,2,4,7,8,13], idx = [3,1,1,0]$$

… 

Loop until we find the $$n$$-th SUN. Time Complexity $$O(n*m)$$ 

### Python code

```python
    def nthSuperUglyNumber(self, n, primes):
        ugly = [1] * n
        factors = primes[:]
        idx = [0] * len(primes)
        for i in range(1, n):
            ugly[i] = min(factors)
            for j in range(len(primes)):
                if factors[j] == ugly[i]:
                    idx[j] += 1
                    factors[j] = primes[j] * ugly[idx[j]]
        return ugly[-1]

```





