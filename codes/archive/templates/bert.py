import os 
import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
import tensorflow_hub as hub
import sklearn.metrics 
import sklearn.model_selection
from absl import logging
logging.set_verbosity(logging.INFO)

# !pip install sentencepiece

# in case there is no such file in local path
if not os.path.exists('tokenization.py'):
    os.system('wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py')
import tokenization

class OverwriteLog():
    # overwrite logging for kaggle kernel
    def info(self, msg):
        print(msg)

logging = OverwriteLog()

def load_data():
    csv = '../input/sms-spam-collection-dataset/spam.csv'
    train = pd.read_csv(csv, encoding = 'Windows-1252')
    train['text'] = train.v2
    train['target'] = (train.v1 == 'spam').astype(int)
    return train
  

def get_bert_layer():
    module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
    module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
    return hub.KerasLayer(module_url, trainable=True)
    

def get_tokenizer(bert_layer):
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    return tokenization.FullTokenizer(vocab_file, do_lower_case)


def bert_encode(texts, tokenizer, max_len=512):
    # encode texts 
    tokens, masks, segments = [], [], []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        cur_tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        cur_tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        tokens.append(cur_tokens)
        masks.append(pad_masks)
        segments.append(segment_ids)
    
    return np.array(tokens), np.array(masks), np.array(segments)


def build_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(net)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model 

"""split the following codes into several chunks in Jyputer 
for clearer reading and saved variables
"""
train = load_data()
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(train.text.values, train.target, test_size=0.2, random_state=0)
logging.info("Data loaded and split")

max_len = 120
bert_layer = get_bert_layer()
tokenizer = get_tokenizer(bert_layer)
logging.info("bert_layer and tokenizer built")

X_train = bert_encode(X_train, tokenizer, max_len=max_len)
X_val = bert_encode(X_val, tokenizer, max_len=max_len)
logging.info("Text tokenized")

# Build model 
model = build_model(bert_layer, max_len=max_len)
model.summary()
logging.info("Model built")

# Run model
checkpoint = tf.keras.callbacks.ModelCheckpoint('bert.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=1,
    callbacks=[checkpoint, earlystopping],
    batch_size=16,
    verbose=1
)
logging.info("Model trainning complete")

# validation & predict
model.load_weights('bert.h5')
y_preds = model.predict(X_val).round().astype(int)
print("Validation accuracy  score", sklearn.metrics.accuracy_score(y_preds, y_val))

