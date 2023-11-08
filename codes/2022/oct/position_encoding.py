#!/usr/bin/env python
import random
import re, os, sys, joblib
from collections import defaultdict
from functools import reduce
import codefast as cf

import numpy as np
import matplotlib.pyplot as plt


def get_position_encoding(seq_len:int, d:int, n=10000):
    """ Get position encoding for a sequence of length seq_len and embedding dimension d.
    """
    pe = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            pe[k, 2 * i] = np.sin(k / denominator)
            pe[k, 2 * i + 1] = np.cos(k / denominator)
    return pe


pe = get_position_encoding(seq_len=4, d=10, n=100)
print(pe)

