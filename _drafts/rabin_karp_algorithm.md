---
layout:     post
title:      Rabin-Karp Algorithm
date:       2020-06-20
tags: [rabin-karp, string search, algorithm]
categories: 
- algorithm
---


[Karp–Rabin algorithm](https://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm) is a string-searching algorithm created by Richard M. Karp and Michael O. Rabin (1987) that uses hashing to find an exact match of a pattern string in a text. More specifically, it uses a [rolling hash](https://en.wikipedia.org/wiki/Rolling_hash) to quickly filter out positions of the text that can not match the pattern, and then checks for a match at the remaining positions. Generalizations of the same idea can be used to find more than one match of a single pattern, or to find matches for more than one pattern.


---
# A bird-view
The algorithm pseudocode
```python
1 def rabin_karp(s[1..n]: string, pattern[1..m]:string)
2     hpattern := hash(pattern[1..m])
3     for i in range(1, n-m+2)
4        hs = hash(s[i..i+m-1])
5        if hs = hpattern
6            if s[i..i+m-1] = pattern[1..m]
7                return i
8    return not_found
```

Lines 2, 4, and 6 each require `O(m)` time. However, line 2 is only executed once, and line 6 is only executed if the hash values match, which is unlikely to happen more than a few times. Line 5 is executed `O(n)` times, but each comparison only requires constant time, so its impact is `O(n)`. The issue is line 4.

To compute the hash value of substring `s[i..i+m-1]`, a naive method requires `O(m)` time if the value is calculated by examining each character. Then an algorithm with a naive hash computation requires `O(mn)` time, the same complexity as a straightforward string matching algorithms. 

To speed up the algorithm, the hash must be computed in **constant time**. The trick is tos reuse the previously calculated hash value of `s[i+1..i+m-1]` when computing the next hash value of `s[i+1..i+m]`, which is possible with a (special) **rolling hash**, a hash function specially designed to enable this operation.

## Trivial Hash Function Won't Work

A trivial rolling hash function just adds the values of each character in the substring, e.g., `h(s[0..m-1]) = sum(h(s[0]) + ... + h(s[m]))`. But such functions preform poorly and produce the same value for many substrings, e.g., `h(ab) = h(ba)`, and therefore result in many comparisons shown on line 6.

Each comparison requires `O(m)` time to do character-to-character examination, the whole algorithm then takes a worst-case `O(mn)` time.

---

# More Sophisticated Rolling Hash

The [Rabin fingerprinting scheme](https://en.wikipedia.org/wiki/Rabin_fingerprint) conducts rehashing with the following formula:

$$ H(s[i+1..i+n]) = ( ( ( H( s[i..n] ) - s[i] * base ) * d ) + s[i+n] + q) \% q $$

where 
* `s` is the text string.
* `q` is a prime modulus.
* `n` is the length of pattern string.
* `base`, base offset. 
    * This method treats every substring as a number in some base, therefore, `base` should be large enough to avoid frequent hash collision. Usually be $$ d^ {(n-1)} $$, where $$ d = 256 $$ is the number of characters in the alphabet.

This function reduce the number of hash value collisions of strings with same counts of characters, e.g., `ab` v.s. `ba`.

<img src='https://i.loli.net/2020/07/05/bqkGNJBm1phjI32.png' width='25px'> Could collisions be completely avoided ? 

<img src='https://i.loli.net/2020/07/05/ku4QMPcK6gdDpLN.png' width='25px'> No, if we're using the following implementations where the output hash values are modulated by a prime number `q` and thus constrained by a finite domain. We can increase this number, but the length of string is unbounded. 


## How the above formula come ?
Let's put the prime modulus aside for a moment, and consider the hash values of strings with some length `n`. 
For better readability, we write $$h(0, i)$$ as the hash value of string `s[0..i]` (inclusive on both ends) and `s[i]` the numeric value of character `s[i]`, then: 

1. $$h(0, 0) = s[0]$$, 
2. $$h(0, 1) = h(0, 0) * d + s[1]$$, 
3. ...
4. $$h(0, n-1) = h(0, n-2) * d + s[n-1]$$, 

Then 

$$\begin{eqnarray}
h(0, n-1) &=& h(0, n - 2) * d + s[n-1] \\
          &=& (h(0, n-3) * d + s[n-2]) * d + s[n-1] \\
          &=& h(0, n-3) * d^2 + s[n-2] * d + s[n-1] \\
          &=& h(0, n-4) * d^3 + s[n-3] * d^2 + s[n-2] * d + s[n-1] \\
          &=& ...\\
          &=& s[0] * d ^ {n-1} + s[1] * d ^ {n-2} + ... + s[n-1]
\end{eqnarray}$$

Similarly, we can calculate

$$\begin{eqnarray}
    h(1, n) &=& s[1] * d ^ {n-1} + ... + s[n-1] * d + s[n]
\end{eqnarray}$$

Combine the above two formula, we have 

$$ h(1, n) = (h(0, n-1) - s[0] * d ^ {n-1}) * d + s[n] $$

The constant value $$d^{n-1}$$ is exactly our base offset.

---

# Implementation 
```python
def rabin_karp(s: string, pat: string):
    '''Implementation of Rabin-Karp Algorithm to find all occurrence
    of pat in s.
    '''
    m, n = len(s), len(pat)
    h_pat = 0  # Hash value for pat
    h_sub = 0  # Hash value for substring (with length n) of s
    prime_modulus = 997  # A prime number serves as modulus
    d = 256  # The number of characters in the input alphabet
    base_offset = 1  # Base offset, which equals pow(d, n-1) % prime_modulus

    # To avoid overflowing integer maximums when the pattern string is longer,
    # the pattern length base offset is pre-calculated in a loop, modulating
    # the result each iteration
    for i in range(n - 1):
        base_offset = (base_offset * d) % prime_modulus

    # Calculate the hash value of pattern and first window of s
    for i in range(n):
        h_pat = (d * h_pat + ord(pat[i])) % prime_modulus
        h_sub = (d * h_sub + ord(s[i])) % prime_modulus

    print(h_pat, h_sub)

    results = [] # Indexes of pat occurring in s 
    for i in range(m - n + 1):
        if h_pat == h_sub and pat == s[i:i + n]:
            results.append(i)
        if i < m - n:
            h_sub = ((h_sub - ord(s[i]) * base_offset) * d +
                     ord(s[i + n]) + prime_modulus) % prime_modulus

    return results
```


---

# Extra

## Spurious Hit
When the hash value of the pattern matches with the hash value of a window of the text but the window is not the actual pattern then it is called a *spurious hit*. This happens because the hash function maps different strings to the same hash value.

Spurious hit increases the time complexity of the algorithm. In order to minimize spurious hit, we use modulus. It greatly reduces the spurious hit.

