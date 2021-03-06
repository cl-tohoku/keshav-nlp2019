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
import pickle

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
    total_dataset = [json.loads(ln) for ln in open(args.test_data)]
    logging.info("# instances: {}".format(len(total_dataset)))

    # Divide them
    total_p, total_c, total_w, labels = [], [], [], []

    for sentence in total_dataset:
        total_p.append(sentence["premise"])
        total_c.append(sentence["claim"])
        total_w.append(sentence["warrant"])
        labels.append(sentence["label"])

    # Padding premise and claim
    t = pickle.load(open("tok.pickle", 'rb'))
    
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

    m = model.create_conneau_model(args, t, max_tokens, args.warrant)
    print(m.summary())

    m.load_weights("model.hdf5")
    
    logging.info("Start evaluation...")

    m.compile(optimizer='adagrad', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    if args.warrant:
        score = m.evaluate([padded_premise, padded_warrant, padded_claim],
                   labels)
        print(score)
    else:
        score = m.evaluate([padded_premise, padded_claim],
                   labels)
        print(score)
    
        
if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-data','--test-data', dest='test_data', required=True,
        help="Test data.")
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
    parser.add_argument(
        '-w','--warrant', dest='warrant', action='store_true',
        help="Use warrant.")
    args = parser.parse_args()
    main(args)
