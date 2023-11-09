import math
import pandas as pd
import numpy as np
import scipy.stats
from itertools import groupby
import collections
import pprint


def calculate_entropy(df, debug=False):
    """ Calculates entropy of the passed `list`
    """
    cter = collections.Counter(d[-1] for d in df).values()
    _sum = sum(cter)
    return sum(- (c * 1.0 / _sum) * math.log(c * 1.0 / _sum, 2) for c in cter)


def split_dataset(df, axis):
    """ Split data set by the passed feature
    Return: [[]] -> list of list
    """
    df.sort(key=lambda x: x[axis])
    return ([e[0:axis] + e[axis + 1:] for e in g]
            for _, g in groupby(df, lambda x: x[axis]))


def select_best_feature(ds, debug=False):
    """ Find the index of the best feature that produces lowest entropy
    Return: int
    """
    best_feature, entropy_lowest = 0, 0x3f3f3f3f

    for axis in range(len(ds[0]) - 1):
        ds_reduced = split_dataset(ds, axis)
        cur_entropy = sum(map(lambda d: calculate_entropy(d)
                              * (len(d) / len(ds)), ds_reduced))
        if cur_entropy < entropy_lowest:
            entropy_lowest = cur_entropy
            best_feature = axis

    return best_feature


def majority_vote(classlist):
    """ Vote for the majority class of a list
    """
    return max(set(classlist), key=classlist.count)


def grow_tree(dataset, names):
    """ Recursively grow a decision tree by selecting best feature at each step
    Return: dict
    """
    labels = [s[-1] for s in dataset]
    if len(set(labels)) == 1:        # all labels are the same
        return labels[0]

    # NO features were left, but labels are not unique
    if len(dataset[0]) == 1:
        return majority_vote(labels)

    best_feature = select_best_feature(dataset)
    feature_values = sorted(list(set(s[best_feature] for s in dataset)))
    reduced_datasets = [list(e) for e in split_dataset(dataset, best_feature)]

    fn = names[best_feature]
    root = {fn: {}}
    del(names[best_feature])

    for fv, rds in zip(feature_values, reduced_datasets):
        print("Feature value and sub DS", fv, rds)
        root[fn][fv] = grow_tree(rds, names[:])

    return root

if __name__ == "__main__":
    data = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    feature_names = ['no surfacing', 'flippers']
    print('Entropy', calculate_entropy(data))
    r = grow_tree(data, feature_names)
    pprint.pprint(r)


    lenses = [i.strip().split(' ')[1:] for i in open('lenses.data', 'r').readlines()]
    for i in range(len(lenses)):
        lenses[i] = [int(e) for e in lenses[i] if e]
    
    print('Head of lenses', lenses[:10])

    names = ['age', 'prescript', 'astigmatic', 'tearRate']
    r = grow_tree(lenses, names)
    pprint.pprint(r)

