import os
from collections import defaultdict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv, sys, nltk
import numpy as np
from numpy import zeros
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation,Embedding, Flatten, Dense, LSTM, Multiply, Input, Dot
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import itertools

embeddings_index = {}
premises = []
claim = []
warrant_1 = []

with open('/home/keshavsingh/glove.6B.100d.txt','r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
f.close()
#print('Total number of word embeddings found are: %s' % len(embeddings_index))
#print(embeddings_index.get('the',None))

#Premise,claim extraction
with open('/home/keshavsingh/train.tsv','r') as f:
    for line in csv.reader(f, delimiter='\t'):
        tokenized_premise = [x.lower() for x in nltk.word_tokenize(line[4])]
        tokenized_claim = [x.lower() for x in nltk.word_tokenize(line[5])]
        premises.append(tokenized_premise)
        claim.append(tokenized_claim)

# all the vocabulary in premise and claim
max_seqs = (max(len(prem) for prem in premises))
vocab_cp = [c+p for c,p in zip(claim, premises)]
#vocab_total = (set(list(itertools.chain(*(vocab_cp)))))

#Creating dictionary
d = defaultdict(list)
for i,e in enumerate(premises):
    d[i]=e
print(d[0])

#Creating embeddings for Premise
t = Tokenizer()
t.fit_on_texts(vocab_cp)
vocab_size = len(t.word_index) + 1
encoded_docs_premise = t.texts_to_sequences(premises)
encoded_docs_claim = t.texts_to_sequences(claim)
padded_docs_premise = pad_sequences(encoded_docs_premise, maxlen=max_seqs, padding='post')
padded_docs_claim = pad_sequences(encoded_docs_claim, maxlen=max_seqs, padding='post')
#print(encoded_docs_claim)

#vector respresentation of every word in vocab_cp
embedding_matrix = zeros((vocab_size, 100))
for word,i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
#print(embedding_matrix.shape)
#print(embedding_matrix)

# Making model with Keras functional API

input_premise = Input(shape = (max_seqs,), dtype='int32')
input_claim = Input(shape = (max_seqs,), dtype='int32')

#Creating embedding for the input sequence
emb = Embedding(output_dim = 100, input_dim = vocab_size, weights=[embedding_matrix], input_length = max_seqs)

x_premise = emb(input_premise)
x_claim = emb(input_claim)

# LSTM to transform into single vector
lstm_out_premise = LSTM(32)(x_premise)
lstm_out_claim = LSTM(32)(x_claim)
y = Multiply()([lstm_out_premise, lstm_out_claim])
output = Dense(2, activation='softmax')(y)
#y = Activation('softmax')(y)
model = Model(inputs=[input_premise, input_claim], outputs=[output])
#print(model.summary())

#plot_model(model, to_file='claim_premise.png')
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.fit([padded_docs_premise, padded_docs_claim, ], epochs = 80, batch_size = 16, verbose = 1)

#score = model.evaluate([dev_padded_docs_p, dev_padded_docs_c, dev_padded_docs_w1], dev_labels, batch_size=16, verbose=0)
#print(score)


