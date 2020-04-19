---
layout:     post
title:      KMP pattern searching algorithm 
date:       2020-04-18
tags: [kmp, algorithm]
categories: 
- algorithm
---

To search a pattern `pat` (type: string, of length n) from another string `text` (length m), a naive method iterates through each index `i` of `text` and compare `pat == text[i:i+n]?`. The worst case time complexity of this method is $$O(m \cdot (n-m))$$.

A better solution is **KMP(Knuth Morris Pratt)** Pattern Searching, which runs in $$O(n)$$. This method involves two steps: 
1. compute the longest prefix-suffix length list `lps`; 
2. search. 

The idea of using `lps` in the algorithm is to **avoid matching a character that we know will anyway match**, and thus reduce the number of comparison. Formally, 

$$lps_\text{pat}[i] = \text{max}\{ len(u): \exists p, q \ s.t., u = \text{pat}[0..p] \wedge \text{pat[q..i]} = u \mid p \in [0, i), q \in [1, i] \}$$, where `pat[0..p]` includes both `pat[0]` and `pat[p]`. 


# Idea and Pseudo-code
1. let $$m, n$$ be the length of `text`, `pat`, respectively; $$i, j$$ be indexes
2. compare pat[j] == text[i], if:
    1. Yes: then increment i and j by 1
    2. No: set `j = lps[j-1]` and do the comparison again. If `j == 0`, then increment i by 1, go back to comparison

2.2 actually shows the key difference between KMP and a naive algorithm. When a mismatch `pat[j] != text[i]` happens, KMP does not set `i = i - j`, instead, it sets j back to an index j_pre such that `pat[0..j_pre] == text[i-j_pre..i-1]` could be possible. 

This index j_pre is `lps[j-1]`, which is the length of the longest proper prefix-suffix of `pat[0..j-1]`. Then the question is: **why `lps[j-1]` ? why not `lps[j-2]` or something else?**

Answer: we know that: a). `lps[j-1]` is count of characters of `pat[0…j-1]` that are both proper prefix and suffix, and b). `pat[0..j-1] == text[i-j..i-1]`. We can conclude that we do not need to match these `pat[j-1]` characters with `text[i-j…i-1]` because we know that these characters will anyway match. 

## An example 
```bash
pat = abcdabx 
text = abcdabywooduoodu
```
`lps_pat = [0, 0, 0, 0, 1, 2, 0]`, two strings differ at `i=6, j=6`. So we don't have to set `i=1, j=0` to restart the comparison procedure, since we know `pat[0..lps[j-1]] = pat[0..2]` (pat[2] not included) equals `text[4..6]` (text[6] not included). Instead, we should only set `j = lps[5] = 2` and compare `pat[j]` and `text[i]`.

# Implementation 
```python
class KMP():
    def get_lps(self, t: str) -> List[int]:
        '''Compute the length of longest proper prefix-suffix for each t[:i], i \in [1..len(t)],
        where a prefix-suffix of t is a substring, u, of t s.t., t.startswith(u) and t.endswith(u).
        And proper means, len(u) < len(t), i.e., u != t
        '''
        n, i, pre_len = len(t), 1, 0
        lps = [0] * n
        while i < n:
            if t[i] == t[pre_len]:
                pre_len += 1
                lps[i] = pre_len
                i += 1
            else:
                if pre_len == 0:
                    lps[i] = 0
                    i += 1
                else:
                    pre_len = lps[pre_len - 1]
                    # Note that we do not increment i here
        return lps

    def pattern_search(self, s: str, t: str) -> List[int]:
        """KMP (Knuth Morris Pratt) Pattern Searching
        Return a list of indexes i, such that t occurs in s starting from i.
        """
        i, j, m, n = 0, 0, len(s), len(t)
        lps = self.get_lps(t)
        res = []
        while i < m:
            if s[i] == t[j]:
                i, j = i + 1, j + 1
            if j == n:
                res.append(i - n)
                j = lps[j - 1]

            elif i < m and s[i] != t[j]:
                if j == 0:
                    i += 1
                else:
                    j = lps[j - 1]
        return res
```


# References
* [Explain KMP algorithm - Geeksforgeeks](https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/)
