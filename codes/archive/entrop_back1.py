import math
import pandas as pd
import scipy.stats
from itertools import groupby


def calculate_entropy(lis):
    """Calculates entropy of the passed `list`
    """
    data = pd.Series(lis)
    p_data = data.value_counts()           # counts occurrence of each value
    print(p_data)
    entropy = scipy.stats.entropy(p_data)  # get entropy from counts
    return entropy


def split_dataset(lis, axis):
    """Split data set by the passed feature
    Return: [[]] -> list of list
    """
    return [[e[0:axis] + e[axis + 1:] for e in g]
            for _, g in groupby(lis, lambda x: x[axis])]

    # features = set([d[axis] for d in ds])
    # feature_indices = {e: i for i, e in enumerate(list(features))}
    # print(feature_indices)
    # returned_ds = [[] for _ in features]

    # for d in ds:
    #     idx = feature_indices[d[axis]]
    #     returned_ds[idx].append(d[:axis] + d[axis + 1:])

    # return returned_ds


data = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]

# print(map(lambda x: x[:0] + x[1:], data))
# axis = 0
# ds_reduced = [[e[0:axis] + e[axis + 1:] for e in g]
#               for _, g in groupby(data, lambda x: x[axis])]
# print(ds_reduced)

print('Entropy', calculate_entropy(data))
r = split_dataset(data, 0)
print(r)
