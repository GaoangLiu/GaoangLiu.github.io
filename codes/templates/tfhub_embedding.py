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

train, test = load_data()
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(train['text'], train['target'], random_state=0)

model = 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1'
hub_layer = hub.KerasLayer(model, output_shape=[128], input_shape=[], 
                           dtype=tf.string, trainable=True)

is_binary = True 
activation = 'sigmoid' if is_binary else 'softmax'

input = tf.keras.Input(shape=(), name="Input", dtype=tf.string)
net = hub_layer(input)
net = tf.keras.layers.Dense(128)(net)
net = tf.keras.layers.Dense(64)(net)
output = tf.keras.layers.Dense(class_number, activation=activation)(net)
model = tf.keras.models.Model(input, output)
model.summary()


model.compile(
    loss = tf.losses.BinaryCrossentropy(from_logits=True),
    optimizer = 'adam',
    metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy")])

cp = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_weights.hdf5', monitor='val_accuracy', 
    verbose=1, save_best_only=True, mode='max')

# Early stopping 
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto',
    baseline=None, restore_best_weights=False
)

history = model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          batch_size=256,
          callbacks=[cp, es],
          epochs = 30, verbose=1)


model.load_weights('best_weights.hdf5')
y_preds = model.predict(test.text)
logging.info("Prediction DONE")


sub = pd.read_csv('sample.csv')
sub['sentiment'] = y_preds
sub.to_csv('submission.csv', index=False)
# !curl -X PUT --upload-file popcorn_submission.csv ali.140714.xyz:8000/
