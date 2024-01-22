---
layout: post
title: Dynamic Programming Examples
date: 2020-09-01
tags: dynamic_programming
categories: algorithm
author: GaoangLiu
---
* content
{:toc}


"Those who can not remember the past are condemned to repeat it." 






动态规划的思想很简单(算法未必简单)，对一个给定问题(有重叠子问题和最优子结构性质的问题)，先求解其子问题，再根据子问题的解得出原问题的解。拆分什么样的子问题是求解关键。

问题一般有两种求解方式:
1. 自顶向下的备忘录方法
2. 自底向上的方法

第一类方法本质上还是递归求解，但使用数组或者 `map` 存储中间计算结果以避免大量的重复计算，从而节省计算时间。但递归过程中仍然会产生额外的开销。

第二类方法在思想上刚好与第一类相反，与数学归纳法想法类似。先求出最基本子问题的解，然后迭代逐步求解更一般性问题的解。 


## Classic Problem 1: Stone Game
问题描述: [stone game](https://leetcode.com/problems/stone-game/) 考察对一排带有点数的石头序列，比如 [2,3,5,7]，两个玩家依次从序列任意一端取石头并获得相应的点数。假设二人都按最优方式来玩游戏，第一个玩家是否一定可以获胜(取得更多的点数)？

例 1: [1, 3, 5]，第一个玩家可以获胜，先取 5 再取 1 

例 2: [2, 24, 7]，第一个玩家无法获胜

### 自顶向下 + 备忘录方法
```cpp
int memo[501][501];
int dfs(int i, int j, vector<int> &piles){
    if (i == j) return piles[i];
    if (memo[i][j] > 0) return memo[i][j];
    returm (memo[i][j] = max(piles[i] - dfs(i + 1, j, piles), piles[j] - dfs(i, j - 1, piles)));
}

bool stoneGame(vector<int>& piles) {
    memset(memo, 0, sizeof(memo));
    return dfs(0, piles.size() - 1, piles) > 0; 
}
```
时间复杂度 $$O(n^2)$$，空间复杂度 $$O(n^2)$$。其中 501 假设为 `piles` 长度的上界，当上界不存在时，可以使用 `vector` 来替代数组，空间大小为 $$n^2$$ 。


### 自底向上
1. 将其视为零和游戏，求解最后获得的点数比另一个玩家多多少。 
2. 子问题为: 求解对任意长度 `len` 的序列，先手玩家最终的收益。
    * 当序列长度 `len = 1` 时，收益即为石头的点数
    * 对于子序列 `[i, j]`，先手玩家的收益为 `dp(i, j) = max(piles[i] - dp(i+1, j), piles[j] - dp(i, j-1)`

### Algorithm
```cpp
bool stone_game(vector<int> &piles) {
    int n = piles.size(); 
    std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));
    for (int len=1; len<n; len++) 
        for (int i=0; i< n - len; i++) {
            dp[i][i+len] = max(piles[i+len] - dp[i][i+len-1], piles[i] - dp[i+1][i+len]);
        }
    return dp[0].back() > 0; 
}
```
时间复杂度 $$O(n^2)$$，空间复杂度 $$O(n^2)$$

空间可以进一步优化，第2步循环考察的是从下标 `i` 开始，对于固定长度为 `len` 的序列先手玩家的收益。因此可以考虑用列表存储上一步的结果，即从 `i` 开始，对于固定长度为 `len - 1` 的序列先手玩家可获取的收益。

```cpp
bool stone_game(vector<int> &piles) {
    int n = piles.size(); 
    std::vector<int> dp = piles, predp = piles; 
    for (int len=1; len<n; len++) {
        for (int i=0; i< n - len; i++) {
            dp[i] = std::max(piles[i+len] - predp[i], piles[i] - predp[i+1]);
        }
        predp = dp; 
    }
    return dp[0] >= 0;
}
```
再仔细观察，可以看到在更新 `dp` 的时候，用到的值为 `predp[i], predp[i+1]` 而没有用到之前的值，因此可以考虑只用一个列表 `dp` 来存储数据，并将第2个循环中的语句可改写为

```cpp
    dp[i] = std::max(piles[i+len] - dp[i], piles[i] - dp[i+1]);
```


## 最长有效括号
问题[Leetcode 32](https://leetcode.com/problems/longest-valid-parentheses/): 给定一个仅由小括号 `(`, `)` 构成的字符串 `s`，求 `s` 中最长有效子串的长度。比如 `()(()` 最长有效子串为 `()`，长度为2。

思路：构造数列 `dp`，令`dp[i]` 表示以`s[i]`结尾的字符串最长有效子串长度，考虑
1. `s[i]=='('`，则不可能构成有效字符串，这时`dp[i]=0`
2. `s[i] == ')'`，考察
    - `s[i-1] == '('`，那么`dp[i] = 2 + dp[i-2]` (暂不考虑边界条件)
    - `s[i-1] == ')'` 并且 `s[i-dp[i-1]-1] == '('` ，那么`dp[i] = dp[i-1] + 2 + dp[i-dp[i-1]-2]`

举个栗子，`s=()(())`，当前`i=5`,那么 `dp=[0,2,0,0,2,0]`。考虑到有 `s[4] = ')'` 且 `s[5-dp[4]-1] = s[2] = '('`，那么 `dp[5] = dp[4] + 2 + dp[1] = 6`. 

### C++代码

```cpp
class Solution {
public:
  int longestValidParentheses(string s) {
    int res = 0, n = s.size();
    vector<int> dp(n, 0);
    for (int i = 1; i < n; ++i) {
      if (s[i] == ')') {
        if (s[i - 1] == '(') {
          dp[i] = i >= 2 ? 2 + dp[i - 2] : 2;
          res = max(res, dp[i]);
        } else { // s[i-1]==')'
          if (i - dp[i - 1] - 1 >= 0 && s[i - dp[i - 1] - 1] == '(') {
            dp[i] = 2 + dp[i - 1] + (i - dp[i - 1] - 2 >= 0 ? dp[i - dp[i - 1] - 2] : 0);
            res = max(res, dp[i]);
          }
        }
      }
    }
    return res;
  }
};
```
复杂度:
- Time, $$O(n) $$
- Space, $$ O(n) $$


