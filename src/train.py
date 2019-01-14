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
from keras.layers import Concatenate,Activation,Embedding, Flatten, Dense, LSTM, Multiply, Input, Dot, Dropout, Bidirectional, Lambda, Subtract
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
import tensorflow as tf
import itertools

from sklearn.model_selection import StratifiedShuffleSplit

import json
import argparse
import itertools

import model

import logging
logging.basicConfig(level=logging.DEBUG)

# To reduce memory consumption    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def main(args):
    logging.info("Parameters: \n  {}".format("\n".join(["{}={}".format(k, v) for k, v in args.__dict__.items()])))
    logging.info("")
    logging.info("Loading data...")
    
    # Load benchmark.
    total_dataset = [json.loads(ln) for ln in open(args.train_data)]
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
    
    logging.info("vocab size: {}".format(len(t.word_index)))
    encoded_premise = t.texts_to_sequences(total_p)
    encoded_claim = t.texts_to_sequences(total_c)
    encoded_warrant = t.texts_to_sequences(total_w)
    sent_numbers = [len(s) for s in encoded_premise] + [len(s) for s in encoded_claim] + [len(s) for s in encoded_warrant]
    max_tokens = max(sent_numbers)

    logging.info("max tokens: {}".format(max_tokens))

    padded_premise = pad_sequences(encoded_premise, maxlen=max_tokens, padding='post')
    padded_claim = pad_sequences(encoded_claim, maxlen=max_tokens, padding='post')
    padded_warrant = pad_sequences(encoded_warrant, maxlen=max_tokens, padding='post')

    m = model.create_conneau_model(args, t, max_tokens)
    print(m.summary())
    
    #plot_model(model, to_file='claim_premise.png')
    m.compile(optimizer='adagrad', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    logging.info("Start training...")

    es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=15,
                                  verbose=1, mode='auto')

    # Use stratified split so that class distribution in training data and validation data is the same
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    data = np.array(list(zip(padded_premise, padded_warrant, padded_claim)))
    labels = np.array(labels)
    
    for tr, vl in sss.split(data, labels):
        train_X, train_Y = list(zip(*data[tr])), labels[tr]
        val_X, val_Y = list(zip(*data[vl])), labels[vl]

        train_X = list(map(lambda x: np.array(x), train_X))
        val_X = list(map(lambda x: np.array(x), val_X))

        m.fit(
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
    parser.add_argument(
        '-mlp','--mlp-layers', dest='mp_mlp_layers', default=2,
        help="Layers of MLP.")
    parser.add_argument(
        '-mlp-do','--mlp-dropout', dest='mp_mlp_dropout', default=0.5,
        help="Dropout rate of MLP.")
    parser.add_argument(
        '-mlp-dim','--mlp-dim', dest='mp_mlp_dim', default=100,
        help="Hidden dim of MLP.")
    parser.add_argument(
        '-sentenc-dim','--sentencoder-dim', dest='mp_sentenc_dim', default=100,
        help="Hidden dim of sentence encoder.")
    
    args = parser.parse_args()
    main(args)