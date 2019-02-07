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

import logging
logging.basicConfig(level=logging.DEBUG)


def mot():
    return Lambda(lambda x: keras.backend.mean(x, axis=1))

def maxp():
    return Lambda(lambda x: keras.backend.max(x, axis=1))

def kabs():
    return Lambda(lambda x: keras.backend.abs(x))


def load_pretrained_emb(t, fn = '/home/keshavsingh/Glove_embeddings/glove.6B.100d.txt'):
    vocab_size = len(t.word_index) +1
    embeddings_index = {}

    # Word embeddings extraction
    with open(fn, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Embedding matrix
    embedding_matrix = zeros((vocab_size, 100))
    hit = 0
    
    for word,i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            hit += 1
            embedding_matrix[i] = embedding_vector
            
        else:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), 100)

    logging.info("Hit: {}".format(hit))
    
    return embedding_matrix


def create_conneau_model(args, t, max_tokens, use_warrant):
    sentences = None
    inputs = None
    vocab_size = len(t.word_index) +1
    embedding_matrix = load_pretrained_emb(t)
            
    input_premise = Input(shape = (max_tokens,), dtype='int32')
    input_claim = Input(shape = (max_tokens,), dtype='int32')
    input_warrant = Input(shape = (max_tokens,), dtype='int32')
    emb = Embedding(output_dim = 100, input_dim = vocab_size, weights=[embedding_matrix], input_length = max_tokens, mask_zero=True)

    x_premise = emb(input_premise)
    x_claim = emb(input_claim)
    x_warrant = emb(input_warrant)
    enc = Bidirectional(LSTM(args.mp_sentenc_dim, return_sequences=True))
    
    lstm_out_premise = maxp()(enc(x_premise))
    lstm_out_claim = maxp()(enc(x_claim))
    lstm_out_warrant = maxp()(enc(x_warrant))
    
    #
    # Feature extraction
    if use_warrant:
        sentences = [lstm_out_premise, lstm_out_warrant, lstm_out_claim]
        inputs=[input_premise, input_warrant, input_claim]
    else:
        sentences = [lstm_out_premise, lstm_out_claim]
        inputs=[input_premise, input_claim]
    features = []
    features += sentences
    
    # Capture interaction between sentences (Conneau EMNLP2017)
    for oc, op in itertools.combinations(sentences, 2):
        minus = kabs()(Subtract()([oc, op]))
        mul = Multiply()([oc, op])
        features += [minus, mul]
        
    features += [Multiply()(sentences)]
    
    y = Concatenate()(features)
    
    # Hidden layers.
    for i in range(args.mp_mlp_layers):
        y = Dropout(args.mp_mlp_dropout)(y)
        y = Dense(args.mp_mlp_dim, activation="tanh")(y)
    
    y = Dropout(args.mp_mlp_dropout)(y)
    output = Dense(2, activation='softmax')(y)
    
    return Model(inputs=inputs, outputs=[output])
    
