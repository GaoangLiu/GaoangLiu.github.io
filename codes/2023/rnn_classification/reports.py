# --------------------------------------------
import codefast as cf
import joblib
import numpy as np
import pandas as pd
import premium as pm
from codefast.patterns.pipeline import BeeMaxin, Pipeline
from codefast.text import MarkDownHelper
from pydantic import BaseModel
from rich import print
# —--------------------------------------------


def rank_rnn():
    data = []
    js = []
    for f in cf.io.walk('/tmp/rnn'):
        js.append(cf.js(f))
        
    js.sort(key=lambda x: x['f1-score'], reverse=True)
    for j in js:
        data.append(
            [j['max_length'], j['num_layers'], j['bidirectional'],
                round(j['f1-score'], 4), round(j['precision'], 4),
                round(j['recall'], 4)])
               
    header = [
        'max_length', 'num_layers', 'bidirectional', 'f1-score', 'precision',
        'recall'
    ]
    print(MarkDownHelper.to_table(header, data))

if __name__ == '__main__':
    rank_rnn()
