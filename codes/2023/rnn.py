#!/usr/bin/env python3
import numpy as np
import pandas as pd

# create RNN architecture
learning_rate = 0.0001
seq_len = 50
max_epochs = 25
hidden_dim = 100
output_dim = 1
bptt_truncate = 5     # backprop through time --> lasts 5 iterations
min_clip_val = -10
max_clip_val = 10

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

