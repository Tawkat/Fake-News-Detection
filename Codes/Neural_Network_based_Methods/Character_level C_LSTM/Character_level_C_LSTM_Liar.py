import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

import tensorflow as tf
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Lambda
from keras.layers import Conv1D,Conv2D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed,RepeatVector
from keras.models import Model,Sequential
from keras.regularizers import l2

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from sklearn.preprocessing import LabelEncoder

import nltk
nltk.download('punkt')



data_train = pd.read_csv('train.csv')
print(data_train.shape)

data_test = pd.read_csv('test.csv')
print(data_test.shape)


MAX_SENT_LENGTH = 300
MAX_SENTS = 20
MAX_NB_WORDS = 400000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

from nltk import tokenize

def striphtml(html):
    p = re.compile(r'<.*?>')
    return p.sub('', html)


def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)

docs = []
labels = []
texts = []
txt=''

for statement, label in zip(data_train.Statement, data_train.Label):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', clean(striphtml(statement)))
    sentences = [sent.lower() for sent in sentences]
    docs.append(sentences)
    labels.append(label)

for doc in docs:
    for s in doc:
        txt += s
chars = set(txt)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


X = np.ones((len(docs), MAX_SENTS, MAX_SENT_LENGTH), dtype=np.int64) * -1
y = np.array(labels)

for i, doc in enumerate(docs):
    for j, sentence in enumerate(doc):
        if j < MAX_SENTS:
            for t, char in enumerate(sentence[-MAX_SENT_LENGTH:]):
                X[i, j, (MAX_SENT_LENGTH-1-t)] = char_indices[char]


docs_test = []
labels_test = []
texts_test = []
txt=''

for statement, label in zip(data_test.Statement, data_test.Label):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', clean(striphtml(statement)))
    sentences = [sent.lower() for sent in sentences]
    docs_test.append(sentences)
    labels_test.append(label)


X_test = np.ones((len(docs_test), MAX_SENTS, MAX_SENT_LENGTH), dtype=np.int64) * -1
y_test = np.array(labels_test)

for i, doc in enumerate(docs_test):
    for j, sentence in enumerate(doc):
        if j < MAX_SENTS:
            for t, char in enumerate(sentence[-MAX_SENT_LENGTH:]):
                if char not in char_indices.keys():
                   continue
                X_test[i, j, (MAX_SENT_LENGTH-1-t)] = char_indices[char]


indices = np.arange(X.shape[0])
np.random.shuffle(indices)
data = X[indices]
labels = y[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

x_test=X_test
y_test=labels_test

print('Number of positive and negative News in traing and validation set')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))



filter_length = [ 3, 5]
nb_filter = [128,256]
pool_length = 2

def binarize(x, sz=71):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))
def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 71


in_sentence = Input(shape=(MAX_SENT_LENGTH,), dtype='int64')
# binarize function creates a onehot encoding of each character index
embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)

for i in range(len(nb_filter)):
    embedded = Conv1D(filters=nb_filter[i],
                      kernel_size=filter_length[i],
                      padding='valid',
                    activation='relu',
                      kernel_initializer='glorot_normal',
                      strides=1)(embedded)

    embedded = Dropout(0.1)(embedded)
    embedded = MaxPooling1D(pool_size=pool_length)(embedded)

bi_sent = Bidirectional(LSTM(100, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(embedded)


from keras.layers import Concatenate,BatchNormalization

sent_encode = Dropout(0.3)(bi_sent)

encoder = Model(in_sentence,sent_encode)

sequence = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int64')
encoded = TimeDistributed(encoder)(sequence)
bi_seq = Bidirectional(LSTM(100, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(encoded)

output = Dropout(0.3)(bi_seq)
output = Dense(128, activation='relu')(output)
output = Dropout(0.3)(output)
output = Dense(1, activation='sigmoid')(output)

model = Model(sequence,output)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.summary()


'''from keras.utils.vis_utils import plot_model
plot_model(model, to_file='Merged/model_plot.png', show_shapes=True, show_layer_names=True)'''


model.fit(x_train, y_train, epochs=10, batch_size=128)


score=model.evaluate(x_test,y_test,verbose=1)
print('acc: '+str(score[1]))

from sklearn.metrics import precision_recall_fscore_support,classification_report
y_pred=model.predict(x_test)
#print(y_pred)
y2=[]
for q in y_pred:
  if(q[0]>0.5):
    y2.append(True)
  else:
    y2.append(False)
print('Classification report:\n',classification_report(y_test,y2))
#print('Classification report:\n',precision_recall_fscore_support(y_test,y_pred))
#print(y_pred)




