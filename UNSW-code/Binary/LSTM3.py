from __future__ import print_function
#from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

traindata = pd.read_csv('UNtrain.csv', header=0)
testdata = pd.read_csv('UNtest.csv', header=0)

print(traindata.head())

X = traindata.iloc[0:, 1:43]
print(X.head())
Y = traindata.iloc[0:, 43]
print(Y.head())

C = testdata.iloc[0:, 43]
T = testdata.iloc[0:, 1:43]

print("1.....")

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
#print(trainX[0:5,:])

scaler = Normalizer().fit(T)
testT = scaler.transform(T)
# summarize transformed data
np.set_printoptions(precision=3)
#print(testT[0:5,:])


y_train = np.array(Y)
y_test = np.array(C)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))


batch_size = 32

# 1. define the network
model = Sequential()
model.add(LSTM(32,input_dim=42, return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(LSTM(32, return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(LSTM(32, return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(LSTM(32, return_sequences=False))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="results/LSTM3results/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True,monitor='val_acc',mode='max')
csv_logger = CSVLogger('results/LSTM3results/training_set_iranalysis3.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10, validation_data=(X_test, y_test),callbacks=[checkpointer,csv_logger])
model.save("results/LSTM3results/lstm4layer_model.hdf5")


loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))







