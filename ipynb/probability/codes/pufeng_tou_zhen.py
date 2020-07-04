from concurrent.futures import ThreadPoolExecutor
import os
import random
import math
import numpy as np


def trial(arg=None):
    ''' The probability of a needle with length l will cross either of
    x = 0 or x = d
    '''
    d, l = 10, 5
    y = random.uniform(0, 1) * d
    x = random.randint(1, 10001)
    theta = random.uniform(0, 1) * 2 * np.pi
    # print(theta)
    y_1 = y + l * math.sin(theta)
    # print(y, y_1)
    return 0 if 0 < y_1 < d else 1


if __name__ == "__main__":
    cnt = 0
    N = 100_000
    with ThreadPoolExecutor(max_workers=50) as e:
        results = e.map(trial, range(N))
        for f in results:
            cnt += f
    print(cnt)
    print("Estimated pi is ", 2 * 5 * N / (10 * cnt))
