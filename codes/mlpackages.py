# General package
import warnings
warnings.filterwarnings('ignore')

import math, re, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import smart_open

# sklearn packages
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Text processing
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS


# To be classified 
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential 
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, LSTM
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
