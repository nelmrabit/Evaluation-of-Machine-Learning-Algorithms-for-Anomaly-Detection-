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
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger


data = pd.read_csv('CICIDS2017-full.csv', header=0)



print(data.head())

X = data.iloc[0:, 0:78]
print(X.head())
Y = data.iloc[0:, 78]
print(Y.head())

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.3)
print("1.....")

scaler = Normalizer().fit(Xtrain)   # train
trainX = scaler.transform(Xtrain)

scaler = Normalizer().fit(Xtest)
testT = scaler.transform(Xtest)

y_train1 = np.array(Ytrain)
y_test1 = np.array(Ytest)

y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)

X_train = np.array(trainX)
X_test = np.array(testT)



batch_size = 64

# 1. define the network
model = Sequential()
model.add(Dense(1024,input_dim=78,activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(768,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(512,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(256,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(128,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(8))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="results/dnnresults/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('results/dnnresults/training_set_dnnanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train,batch_size=batch_size, nb_epoch=20, callbacks=[checkpointer,csv_logger])
model.save("results/dnnresults/dnn5layer_model.hdf5")


loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
y_pred = model.predict_classes(X_test)







