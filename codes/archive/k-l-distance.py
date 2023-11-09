import numpy as np 

def kldistance(p, q):
    return np.sum(p * (np.log(p) - np.log(q)))

p = np.array([0.4, 0.6])
q = np.array([0.5, 0.5])
print(kldistance(p, q))
print(kldistance(q, p))
