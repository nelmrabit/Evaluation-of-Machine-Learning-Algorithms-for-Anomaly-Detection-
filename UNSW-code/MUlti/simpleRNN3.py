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


traindata = pd.read_csv('UNtrain_multi.csv', header=0)
testdata = pd.read_csv('UNtest_multi.csv', header=0)

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


y_train1 = np.array(Y)
y_test1 = np.array(C)

y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))


batch_size = 32

# 1. define the network
model = Sequential()
model.add(SimpleRNN(32,input_dim=42, return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(SimpleRNN(32, return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(SimpleRNN(32, return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(SimpleRNN(32, return_sequences=False))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="results/simpleRNN3results/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True,monitor='val_acc',mode='max')
csv_logger = CSVLogger('results/simpleRNN3results/training_set_iranalysis3.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=20, validation_data=(X_test, y_test),callbacks=[checkpointer,csv_logger])
model.save("results/simpleRNN3results/lstm4layer_model.hdf5")

loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
y_pred = model.predict_classes(X_test)
#np.savetxt('results/simpleRNN3results/lstm4predicted.txt', np.transpose([y_test,y_pred]), fmt='%01d')






