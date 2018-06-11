import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation,Bidirectional,SpatialDropout1D
from keras.layers.embeddings import Embedding

#Load Pickled
pickle_load=open('pickle_FullTrain_Guard_Nyt_1_100dim.pickle','rb')
X,y_train,embedding_matrix=pickle.load(pickle_load)

embedding_size=100

## create model
model_glove = Sequential()
model_glove.add(Embedding(vocabulary_size, embedding_size, weights=[embedding_matrix], trainable=False))

model_glove.add(LSTM(300))

model_glove.add(Dense(1, activation='sigmoid'))

model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_glove.summary()
## Fit train data
history=model_glove.fit(X, y_train, validation_split=0.2, epochs = 10,batch_size=64,shuffle=True)
