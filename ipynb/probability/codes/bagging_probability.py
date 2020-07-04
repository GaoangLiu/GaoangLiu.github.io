import os
from scipy.special import comb


def bagging_proba(n, acc=0.8):
    ''' n independent estimators with accuracy 'acc', then what is estimated
    accuracy of bagging estimator with majority voting ?
    Note, 6 or more out of 10 is called a majority. 
    '''
    if n == 1:
        return acc
    error = 0
    for i in range(n // 2 + 1):
        # only i estimator makes correct guess
        error += comb(n, i, exact=False) * \
            ((1 - acc) ** (n - i)) * ((acc) ** i)
    return 1 - error


for i in range(1, 10):
    n = i * 10
    print(n, bagging_proba(n))

'''
10 0.9672065024000001
20 0.997405172599326
30 0.9997687743883322
40 0.9999783081068737
50 0.9999979051451444
60 0.9999997938783113
70 0.999999979452253
80 0.999999997931789
90 0.9999999997902754
'''
