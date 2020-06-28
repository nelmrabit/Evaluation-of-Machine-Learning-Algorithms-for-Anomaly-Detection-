from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
from keras.datasets import imdb
from keras import backend as K
#from sklearn.cross_validation import train_test_split
import pandas as pd
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.utils import np_utils
import numpy as np
import h5py
from keras import callbacks
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics


#traindata = pd.read_csv('kdd/kddtrain.csv', header=None)
testdata = pd.read_csv('UNtest.csv', header=0)


#X = traindata.iloc[:,0:42]
#Y = traindata.iloc[:,0]
C = testdata.iloc[0:, 43]
T = testdata.iloc[0:, 1:43]



scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_test = np.array(C)


X_test = np.array(testT)





batch_size = 64

# 1. define the network
model = Sequential()
model.add(Dense(1024,input_dim=42,activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(768,activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# define optimizer and objective, compile cnn
'''
cnn.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])

# train
checkpointer = callbacks.ModelCheckpoint(filepath="results/cnn3results/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = CSVLogger('results/cnn3results/cnntrainanalysis3.csv',separator=',', append=False)
cnn.fit(X_train, y_train, nb_epoch=1000, show_accuracy=True,validation_data=(X_test, y_test),callbacks=[checkpointer,csv_logger])
cnn.save("results/cnn3results/cnn_model.hdf5")
'''
'''
cnn.load_weights("results/cnn3results/checkpoint-317.hdf5")

y_pred = cnn.predict_classes(X_test)

np.savetxt('res/expected3.txt', y_test1, fmt='%01d')
np.savetxt('res/predicted3.txt', y_pred, fmt='%01d')
cnn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = cnn.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
'''

model.load_weights("results/dnnresults/dnn5layer_model.hdf5")
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))


y_pred = model.predict_classes(X_test)


accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred , average="binary")
precision = precision_score(y_test, y_pred , average="binary")
f1 = f1_score(y_test, y_pred, average="binary")

print("confusion matrix")
print("----------------------------------------------")
print("accuracy")
print("%.6f" %accuracy)
print("racall")
print("%.6f" %recall)
print("precision")
print("%.6f" %precision)
print("f1score")
print("%.6f" %f1)
cm = metrics.confusion_matrix(y_test, y_pred)
print("==============================================")
