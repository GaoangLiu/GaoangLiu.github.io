import random
import math
import itertools

def bucket_sort(arr):
    n = len(arr)
    k = int(math.sqrt(n))
    buckets = [[] for _ in range(k)]
    v_min, v_max = min(arr), max(arr)
    divider = (v_max - v_min) // (k - 1)
    for n in arr:
        index = (n - v_min) // divider
        buckets[index].append(n)

    for i in range(k):
        buckets[i].sort()
    return list(itertools.chain(*buckets))


arr = list(range(100))
random.shuffle(arr)
arr = random.choices(arr, k=39)

sorted = bucket_sort(arr)
print(sorted)
