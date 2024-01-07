---
layout: post
title: KMP pattern searching algorithm
date: 2020-04-18
tags: kmp algorithm
categories: algorithm
author: gaonagliu
---
* content
{:toc}


To search a pattern `pat` (type: string, of length n) from another string `text` (length m), a naive method iterates through each index `i` of `text` and compare `pat == text[i:i+n]`. The worst case time complexity of this method is $$O(m \cdot (n-m))$$.




A better solution is [**KMP(Knuth Morris Pratt)**]([https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm](https://en.wikipedia.org/wiki/Knuth–Morris–Pratt_algorithm)) Pattern Searching, which runs in $$O(n)$$. This method involves two steps: 
1. compute the longest proper prefix length list `lhp` of pattern `pat` 
2. search. 

# Part 1: Preprocessing
## Computing longest happy prefix length list 

The **proper prefix**,  also called **happy prefix** in some literatures, is a **non-empty** prefix which is also a suffix (excluding `s` itself) of `s`. 

A naive (brute-force) way to find the longest happy prefix (abbreviated as LHP) of `s` will check each prefix of `s`, and determine whether such prefix is also a suffix, and finally return the one, if there is any, with the maximum length. 

Time complexity of this method is `O(n^2)`, where `n` is the length of `s`.




Now think about this question: what if we have computed the length of LHP of `s[0..i] (0 <= i < n-1)`, can we find out the LHP of `s` within constant time ?

If so, by reduction, we find a linear algorithm to compute LHP. 

And the answer is: Yes, we can compute the LHP of `s` in constant time, given that `lhp[i](0<=i<n-1)`were already computed. 

To illustrate, consider an example where `s = abcdabcb`, the corresponding LHP list is `lhp(s) = [0, 0, 0, 0, 1, 2, 3, N]`, we want to find out what the final `N` is. 

Let's denote `t = s[0..n-2]` as the longest prefix of `s`, and `lhp_s = s[0..j]` as the LHP of `s`. Following the above example, `t` shall be `abcdabc` and its LHP is `abc` with length 3.  Thus the length of `lhp_s` is: 

1. `4`, if `s[3] == s[n-1]`,  (`n == 8`)
2.  some `k <= 3`, otherwise 

The first case is trivial, and `s` has no LHP  longer than 4. For otherwise,  if `s` has a LHP `u` such that `len(u) > 4`, then `u[0..len(u)-2]` is a LHP of `t` and its length is surely larger than 3, a contradiction. 

The second case is a little tricky. We know the length of `lhp_s` is 3 only when  `s[n-1] = s[2]`. But in our case, we find that `s[n-1](b) != s[2](c)`, what should we do next ?

Now consider the following facts on the LHP of `s`:

1. By now, `lhp_s = s[0..j] = s[n-1-j..n-1]` can only be a prefix of `abc,` and thus `s[0..j-1]` is also a prefix of `abc`
2. `s[0..j-1]` is a suffix of `abc` since `s[0..j-1] = s[n-1-j..n-2]`
s
This means the subsequence `s[0..j-1]` is a **happy prefix** of `abc` ! Therefore, `lhp_s`  COULD be the LHP of `abc`. 

One more thing we need to do is checking whether `s[n-1] = s[len(lhp[abc])]`, if true, `lhp(s) = j` and our algorithm ends; if false, we repeat the above procedure until we finally found a non-empty LHP of `s` or nothing, in later case `lhp[s] = 0`.


### Python Code of computing LHP list
Base case: LHP of any single-character string `s` is an empty string `""` with length 0.

```python
def longest_happy_prefix(s: str) -> List[int]:
    j = 0
    lhp = [0] * len(s)
    for i in range(1, len(s)): 
        while j > 0 and s[i] != s[j]:
            j = lhp[j - 1]
        
        if s[i] == s[j]:
            j += 1
            lhp[i] = j
    return lhp
```


# Part 2: searching

The idea of using `lhp` list in the algorithm is to **avoid matching a character that we know will anyway match**, and thus reduce the number of comparison. Formally, 


# Idea behind
1. let $$m, n$$ be the length of `text`, `pat`, respectively, and b`i, j` be indexes
2. compare `pat[j] == text[i]`, if:
    1. `true`, then increment i and j by 1
    2. `false`, set `j = lhp[j-1]` and do the comparison again. If `j == 0`, then increment i by 1, go back to comparison

2.2 tells the key difference between KMP and a naive algorithm. When a mismatch `pat[j] != text[i]` happens, KMP does not set `i = i - j`, instead, it sets j back to an index `lhp[j-1]` such that `pat[0..j_pre] == text[i-j_pre..i-1]` could be possible. 

Then the question is: **why `lhp[j-1]` ? why not `lhp[j-2]` or something else?**

Answer: we know that: a). `lhp[j-1]` is the length of LHP of `pat[0…j-1]`, and b). `pat[0..j-1] == text[i-j..i-1]`. We can conclude that we do not need to match these `pat[j-1]` characters with `text[i-j…i-1]` because we know that these characters will anyway match. 

## An example 
```bash
pat = abcdabx 
text = abcdabywooduoodu
```
`lhp_pat = [0, 0, 0, 0, 1, 2, 0]`, two strings differ at `i=6, j=6`. So we don't have to set `i=1, j=0` to restart the comparison procedure, since we know `pat[0..lhp[j-1]] = pat[0..1]` (inclusive) equals `text[4..5]`. Instead, we should only set `j = lhp[5] = 2` and compare `pat[j]` and `text[i]`.

# Implementation 
```python
class KMP():
    def get_lhp(self, t: str) -> List[int]:
        '''Compute the length of LHP for each t[:i], i \in [1..len(t)],
        where a prefix-suffix of t is a substring, u, of t s.t., t.startswith(u) and t.endswith(u).
        And proper means, len(u) < len(t), i.e., u != t
        '''
        j, lhp = 0, [0] * len(t)
        for i in range(1, len(t)):
            while j > 0 and t[i] != t[j]:
                j = lhp[j-1]
                
            if t[i] == t[j]:
                j += 1
                lhp[i] = j
        return lhp

    def pattern_search(self, text: str, pat: str) -> List[int]:
        """KMP (Knuth Morris Pratt) Pattern Searching
        Return a list of indexes i, such that t occurs in s starting from i.
        """
        j = 0
        lhp, res = self.get_lhp(pat), []
        for i in range(len(text)):
            while j > 0 and text[i] != pat[j]:
                j = lhp[j-1]

            if text[i] == pat[j]:
                j += 1 

            if j == len(pat):
                res.append(i + 1 - len(pat))
                j = lhp[j - 1]
        return res
```


# References
* [Explain KMP algorithm - Geeksforgeeks](https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/)
