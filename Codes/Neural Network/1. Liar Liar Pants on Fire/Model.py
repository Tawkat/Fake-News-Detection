from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras.layers import TimeDistributed
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

total_word=10000

time_step=300

# Reading Dataset
dataset=pd.read_csv('train.csv')
statement=dataset.iloc[:,2:3].values
texts=[]
texts=dataset['Statement'].astype(str).values.tolist()
label=dataset['Label'].astype(int).values.tolist()
y_train=label
y_train=np.reshape(y_train,(-1,1))

#Tokenizing texts
tokenizer_train=Tokenizer(num_words=total_word)
tokenizer_train.fit_on_texts(texts)
encoded_train=tokenizer_train.texts_to_sequences(texts=texts)

vocab_size_train = len(tokenizer_train.word_index) + 1
print(vocab_size_train)

X_train = sequence.pad_sequences(encoded_train, maxlen=time_step,padding='post')

#Reading Glove
f = open('glove.6B.300d.txt',encoding='utf-8')
embeddings_train={}
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_train[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_train))

embedding_size=300

# create a weight matrix for words in training docs
embedding_matrix_train = np.zeros((vocab_size_train, embedding_size))
for word, i in tokenizer_train.word_index.items():
	embedding_vector_train = embeddings_train.get(word)
	if embedding_vector_train is not None:
		embedding_matrix_train[i] = embedding_vector_train




print(vocab_size_train)
print(len(embedding_matrix_train))


# Convolution Param
filter_length = 3
nb_filter = 128
pool_length = 2


#Creating Model
model = Sequential()
model.add(Embedding(vocab_size_train, embedding_size,
                    weights=[embedding_matrix_train],trainable=False))


model.add(Conv1D(filters=nb_filter,
                        kernel_size=filter_length,
                        activation='relu'))
model.add(MaxPooling1D(pool_size=pool_length))

model.add(Flatten())

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=64,epochs=30)


#Pickling to save
with open('cnn1_llpf.pickle','wb') as f:
    pickle.dump(model, f)

'''
pickle_in = open('cnn1_llpf.pickle','rb')
model = pickle.load(pickle_in)
'''
