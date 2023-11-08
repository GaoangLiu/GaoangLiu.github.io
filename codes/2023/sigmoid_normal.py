# --------------------------------------------
import os
import random
import json
import re
import sys
import time
from collections import defaultdict
from functools import reduce

import codefast as cf
import joblib
import numpy as np
import pandas as pd
from rich import print
from typing import List, Union, Callable, Set, Dict, Tuple, Optional, Any
import premium as pm

from pydantic import BaseModel
from codefast.patterns.pipeline import Pipeline, BeeMaxin
# —--------------------------------------------

import matplotlib.pyplot as plt


# draw sigmoid and standard normal distribution function, x range from -10 to 10
x = np.linspace(-10, 10, 1000)
y = 1 / (1 + np.exp(-x))
plt.plot(x, y, label='sigmoid')
plt.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), label='normal')

# z is sampled from standard normal distribution
z = np.random.randn(300)
plt.plot(z, 1 / (1 + np.exp(-z)), '.', alpha=0.1, label='sampled', color='red')

# plt.legend()
# plt.show()
# export to png
plt.savefig('sigmoid_normal.png', dpi=200)
