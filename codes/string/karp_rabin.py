import string


def rabin_karp(s: string, pat: string):
    '''Implementation of Rabin-Karp Algorithm to find all occurrence
    of pat in s.
    '''
    m, n = len(s), len(pat)
    h_pat = 0  # Hash value for pat
    h_sub = 0  # Hash value for substring (with length n) of s
    prime_modulus = 997  # A prime number serves as modulus
    d = 256  # The number of characters in the input alphabet
    base_offset = 1  # Base offset, which equals pow(d, n-1) % q

    # To avoid overflowing integer maximums when the pattern string is longer,
    # the pattern length base offset is pre-calculated in a loop, modulating
    # the result each iteration
    for i in range(n - 1):
        base_offset = (d * base_offset) % prime_modulus

    # Calculate the hash value of pattern and first window of s
    for i in range(n):
        h_pat = (d * h_pat + ord(pat[i])) % prime_modulus
        h_sub = (d * h_sub + ord(s[i])) % prime_modulus

    print(h_pat, h_sub)
    occurrences = []
    for i in range(m - n + 1):
        if h_pat == h_sub:
            print('potentail')
            if pat == s[i:i + n]:
                occurrences.append(i)
        if i < m - n:
            h_sub = ((h_sub - ord(s[i]) * base_offset) * d  + ord(s[i + n]) + prime_modulus) % prime_modulus

    return occurrences


s = 'banana'
t = 'ana'
s, t = 'abba', 'ba'
print(rabin_karp(s, t))
