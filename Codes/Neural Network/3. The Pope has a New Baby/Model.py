from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Conv1D,GRU
from keras.layers import MaxPooling1D,Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras.layers import TimeDistributed
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder


total_word=10000

time_step=200
hidden_size=100
embedding_size=300
dropout=0.5

# Reading Dataset
dataset=pd.read_csv('fake_or_real_news.csv')

texts=dataset['text'].astype(str).values.tolist()

label=dataset['label']


#Tokenizing texts
tokenizer=Tokenizer(num_words=total_word)
tokenizer.fit_on_texts(texts)
encoded=tokenizer.texts_to_sequences(texts=texts)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

X = sequence.pad_sequences(encoded, maxlen=time_step,padding='post')
labelEncoder=LabelEncoder()
label=labelEncoder.fit_transform(label)
y=np.reshape(label,(-1,1))


#Reading Glove
f = open('glove.6B.300d.txt',encoding='utf-8')
embeddings={}
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, embedding_size))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector




X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


print(vocab_size)
print(len(embedding_matrix))
print(embedding_matrix.shape[1])

maxlen = 200



#Creating Model
model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=maxlen,
                    weights=[embedding_matrix],trainable=False))

model.add(GRU(100))




model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64,epochs=30,validation_data=(X_test,y_test))

#score=model.evaluate(X_test,y_test,verbose=1)

