#!/usr/bin/env python
import codefast as cf

from tensorflow import matmul, math, cast, float32
from tensorflow import keras as K
import numpy as np


# Implementing the Scaled-Dot Product Attention
class DotProductAttention(K.layers.Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self,
             queries: np.array,
             keys: np.array,
             values: np.array,
             d_k: int,
             mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(
            cast(d_k, float32))

        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        # Computing the weights by a softmax operation
        weights = K.backend.softmax(scores)

        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)


def test():
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    batch_size = 3  # Batch size from the training process

    input_seq_length = 5  # Maximum length of the input sequence

    queries = np.random.random((batch_size, input_seq_length, d_k))
    keys = np.random.random((batch_size, input_seq_length, d_k))
    values = np.random.random((batch_size, input_seq_length, d_v))
    attention = DotProductAttention()
    print(attention(queries, keys, values, d_k))

if __name__ == '__main__':
    test()
    
