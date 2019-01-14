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

import logging
logging.basicConfig(level=logging.DEBUG)

# To reduce memory consumption    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)
    
embeddings_index = {}
premise = []
claim = []
warrant = []

#Word embeddings extraction
with open('/home/keshavsingh/glove.6B.100d.txt','r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
f.close()

#Premise,claim extraction
with open('/Users/keshavsingh/Desktop/train.tsv','r') as f:
    for line in csv.reader(f, delimiter='\t'):
        t_premise = line[4]
        t_claim = line[5]
        t_label = line[3]
        t_warrant_1 = line[2]
        t_warrant_0 = line[1]
        premise.append(t_premise)
        claim.append(t_claim)
        warrant_0.append(t_warrant_0)
        warrant_1.append(t_warrant_1)
        label.append(t_label)

#Warrant correct extraction
warrants_labels = list(zip(warrant_0,warrant_1,label))
warrant_correct = []
for w0,w1,l in warrants_labels:
    if l == '0':
        w = w0
    else:
        w = w1
    warrant_correct.append(w)

#Generating data (positive and negative)
pos_dataset = list(zip(premise, claim, warrant_correct))
neg_dataset = []
random.seed(123)
for prem, claim, warrant in pos_dataset:
    premise_pool = []

    for p, c, w in pos_dataset:
        if c != claim:
            premise_pool += [p]

    neg_claim = claim
    neg_premise = random.sample(premise_pool, 1)[0]
    neg_dataset += [(neg_premise, neg_claim, warrant)]

#Total dataset
total_dataset = pos_dataset + neg_dataset
labels = [1]*len(pos_dataset) + [0]*len(neg_dataset)

logging.info("# instances: {}".format(len(total_dataset)))


#total (premise and claim)
total_p = []
total_c = []
total_w = []
for sentence in total_dataset:
    total_p.append(sentence[0])
    total_c.append(sentence[1])
    total_w.append(sentence[2])

#Tokenizing & vocabulary
words = []
for sent in total_dataset:
    words += text_to_word_sequence(sent[0])+text_to_word_sequence(sent[1])+text_to_word_sequence(sent[2])
vocab = (set(words))

#Padding premise and claim
t = Tokenizer()
t.fit_on_texts(vocab)
vocab_size = len(t.word_index) +1
encoded_premise = t.texts_to_sequences(total_p)
encoded_claim = t.texts_to_sequences(total_c)
encoded_warrant = t.texts_to_sequences(total_w)
sent_numbers = [len(s) for s in encoded_premise] + [len(s) for s in encoded_claim] + [len(s) for s in encoded_warrant]
max_tokens = max(sent_numbers)

logging.info("max tokens: {}".format(max_tokens))

padded_premise = pad_sequences(encoded_premise, maxlen=max_tokens, padding='post')
padded_claim = pad_sequences(encoded_claim, maxlen=max_tokens, padding='post')
padded_warrant = pad_sequences(encoded_warrant, maxlen=max_tokens, padding='post')
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
emb = Embedding(output_dim = 100, input_dim = vocab_size, weights=[embedding_matrix], input_length = max_tokens)

x_premise = emb(input_premise)
x_claim = emb(input_claim)
x_warrant = emb(input_warrant)
lstm_out_premise = LSTM(32)(x_premise)
lstm_out_claim = LSTM(32)(x_claim)
lstm_out_warrant = LSTM(32)(x_warrant)
y = Concatenate()([lstm_out_premise, lstm_out_warrant, lstm_out_claim])
output = Dense(2, activation='softmax')(y)
model = Model(inputs=[input_premise, input_warrant, input_claim], outputs=[output])
print(model.summary())
#plot_model(model, to_file='claim_premise.png')
model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

logging.info("Start training...")


es = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=4,
                              verbose=1, mode='auto')

model.fit([padded_premise, padded_warrant, padded_claim], labels, epochs = 100, validation_split = 0.1, shuffle = True, callbacks = [es], batch_size = 32, verbose = 1)
