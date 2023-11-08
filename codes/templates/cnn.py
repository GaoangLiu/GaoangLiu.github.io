import os 
import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
import tensorflow_hub as hub
import sklearn
import sklearn.metrics 
import sklearn.model_selection
import sklearn.feature_extraction

from absl import logging
logging.set_verbosity(logging.INFO)

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
  

def vectorize(train, test, vocab_size):
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1,3), token_pattern=r'\w{1,}',
            min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
            smooth_idf=1, sublinear_tf=1, max_features=vocab_size)
    vectorizer.fit(pd.concat([train.text, test.text]))
    train_inputs = vectorizer.transform(train.text)
    test_inputs  = vectorizer.transform(test.text)
    logging.info("Train & test text tokenized")
    return train_inputs, test_inputs

def build_model(vocab_size = 512, max_len = 100, embed_size = 128, embed_matrix=[]):
    text_input = tf.keras.Input(shape=(max_len, ))
    embed_text = tf.keras.layers.Embedding(vocab_size, embed_size)(text_input)
    if len(embed_matrix) > 0:
        embed_text = tf.keras.layers.Embedding(vocab_size, embed_size, \
                                        weights=[embed_matrix], trainable=False)(text_input)
        
    net = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(embed_text)
    net = tf.keras.layers.GlobalMaxPool1D()(net)

    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(net)
    
    model = tf.keras.models.Model(inputs=text_input, outputs=outputs)

    return model

"""split the following codes into several chunks in Jyputer 
for clearer reading and saved variables
"""
vocab_size = 10000
max_len = 120
embed_size = 128


train, test = load_data()
train_inputs, test_inputs = vectorize(train, test)
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(train, train.target, test_size=0.2, random_state=0)
logging.info("Data loaded and split")

# Build model 
model = build_model(vocab_size, max_len, embed_size)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])
model.summary()
logging.info("Model built")

# Run model
checkpoint = tf.keras.callbacks.ModelCheckpoint('best.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=1,
    callbacks=[checkpoint, earlystopping],
    batch_size=1024,
    verbose=1
)
logging.info("Model trainning complete")

# validation & predict
model.load_weights('best.h5')
y_preds = model.predict(X_val).round().astype(int)
print("Validation accuracy  score", sklearn.metrics.accuracy_score(y_preds, y_val))

