#!/usr/bin/env python
import os
import random
import re
import sys
from collections import defaultdict
from functools import reduce

import codefast as cf
import joblib
import matplotlib.pyplot as plt
# Additive white gassusian noise
import numpy as np


def awgn(source: np.ndarray, seed: int = 0, snr: float = 70.0):
    """ snr = 10 * log10( xpower / npower )
    """
    random.seed(seed)
    snr = 10**(snr / 10.0)
    xpower = np.sum(source**2) / len(source)
    npower = xpower / snr
    noise = np.random.normal(scale=np.sqrt(npower), size=source.shape)
    return source + noise


if __name__ == '__main__':
    t = np.linspace(1, 100, 100)
    source = 10 * np.sin(t / (2 * np.pi))
    # four subplots
    f, axarr = plt.subplots(2, 2)
    f.suptitle('Additive white gassusian noise')
    for i, snr in enumerate([10, 20, 30, 70]):
        with_noise = awgn(source, snr=snr)
        axarr[i // 2, i % 2].plot(t, source, 'b', t, with_noise, 'r')
        axarr[i // 2, i % 2].set_title('snr = {}'.format(snr))
    plt.show()
