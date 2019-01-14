import os,random
from random import seed
from collections import defaultdict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv, sys, nltk
import numpy as np
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Concatenate,Activation,Embedding, Flatten, Dense, LSTM, Multiply, Input, Dot
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
import tensorflow as tf
import itertools

from sklearn.model_selection import StratifiedShuffleSplit

import json
import argparse

import logging
logging.basicConfig(level=logging.DEBUG)

# To reduce memory consumption    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def create_model(t, max_tokens):
    vocab_size = len(t.word_index) +1
    embeddings_index = {}

    #Word embeddings extraction
    with open('/home/keshavsingh/glove.6B.100d.txt', 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    #embedding matrix
    embedding_matrix = zeros((vocab_size, 100))
    for word,i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    #Keras Implementation
    input_premise = Input(shape = (max_tokens,), dtype='int32')
    input_claim = Input(shape = (max_tokens,), dtype='int32')
    input_warrant = Input(shape = (max_tokens,), dtype='int32')
    emb = Embedding(output_dim = 100, input_dim = vocab_size, weights=[embedding_matrix], input_length = max_tokens, mask_zero=True)

    x_premise = emb(input_premise)
    x_claim = emb(input_claim)
    x_warrant = emb(input_warrant)
    lstm_out_premise = LSTM(100)(x_premise)
    lstm_out_claim = LSTM(100)(x_claim)
    lstm_out_warrant = LSTM(100)(x_warrant)
    y = Concatenate()([lstm_out_premise, lstm_out_warrant, lstm_out_claim])
    y = Dense(100, activation='tanh')(y)
    output = Dense(2, activation='softmax')(y)
    
    return Model(inputs=[input_premise, input_warrant, input_claim], outputs=[output])
    
    
def main(args):
    # Load benchmark.
    total_dataset = json.load(open(args.train_data))
    logging.info("# instances: {}".format(len(total_dataset)))

    # Divide them
    total_p, total_c, total_w, labels = [], [], [], []

    for sentence in total_dataset:
        total_p.append(sentence["premise"])
        total_c.append(sentence["claim"])
        total_w.append(sentence["warrant"])
        labels.append(sentence["label"])

    # Padding premise and claim
    t = Tokenizer()
    t.fit_on_texts(total_p)
    t.fit_on_texts(total_c)
    t.fit_on_texts(total_w)
    
    encoded_premise = t.texts_to_sequences(total_p)
    encoded_claim = t.texts_to_sequences(total_c)
    encoded_warrant = t.texts_to_sequences(total_w)
    sent_numbers = [len(s) for s in encoded_premise] + [len(s) for s in encoded_claim] + [len(s) for s in encoded_warrant]
    max_tokens = max(sent_numbers)

    logging.info("max tokens: {}".format(max_tokens))

    padded_premise = pad_sequences(encoded_premise, maxlen=max_tokens, padding='post')
    padded_claim = pad_sequences(encoded_claim, maxlen=max_tokens, padding='post')
    padded_warrant = pad_sequences(encoded_warrant, maxlen=max_tokens, padding='post')

    model = create_model(t, max_tokens)
    print(model.summary())
    
    #plot_model(model, to_file='claim_premise.png')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    logging.info("Start training...")


    es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=15,
                                  verbose=1, mode='auto')

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    data = np.array(list(zip(padded_premise, padded_warrant, padded_claim)))
    labels = np.array(labels)
    
    for tr, vl in sss.split(data, labels):
        train_X, train_Y = list(zip(*data[tr])), labels[tr]
        val_X, val_Y = list(zip(*data[vl])), labels[vl]

        train_X = list(map(lambda x: np.array(x), train_X))
        val_X = list(map(lambda x: np.array(x), val_X))

        model.fit(
            train_X,
            train_Y,
            epochs = 100,
            validation_data = (val_X, val_Y),
            callbacks = [es],
            batch_size = 32,
            verbose = 1)

        
if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-data','--train-data', dest='train_data', required=True,
        help="Training data.")

    args = parser.parse_args()
    main(args)