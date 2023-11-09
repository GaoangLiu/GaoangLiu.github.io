import numpy as np
import adaboost 

def load_simple_data():
    data = ([[1., 2.1], [2., 1.1], [1.3], 1.], [1., 1.], [2., 1.])
    labels = [1.0, 1.0, -1.0, -1.0, -1.0]
    return data, labels 

data, labels = load_simple_data()

