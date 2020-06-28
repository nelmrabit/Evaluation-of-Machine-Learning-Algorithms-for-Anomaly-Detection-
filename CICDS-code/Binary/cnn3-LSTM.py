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
from sklearn.model_selection import train_test_split

from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics



data = pd.read_csv('CICIDS2017_binary.csv', header=0)



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

y_train = np.array(Ytrain)   #train gorund truth
y_test = np.array(Ytest)    #test ground truth



# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))




lstm_output_size = 70

cnn = Sequential()
cnn.add(Convolution1D(64, 3, border_mode="same",activation="relu",input_shape=(78, 1)))
cnn.add(Convolution1D(64, 3, border_mode="same", activation="relu"))
cnn.add(MaxPooling1D(pool_length=(2)))
cnn.add(Convolution1D(128, 3, border_mode="same", activation="relu"))
cnn.add(Convolution1D(128, 3, border_mode="same", activation="relu"))
cnn.add(MaxPooling1D(pool_length=(2)))
cnn.add(LSTM(lstm_output_size))
cnn.add(Dropout(0.1))
cnn.add(Dense(1, activation="sigmoid"))

# define optimizer and objective, compile cnn

cnn.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])

# train
checkpointer = callbacks.ModelCheckpoint(filepath="results/cnn3-LSTMresults/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = CSVLogger('results/cnn3-LSTMresults/cnntrainanalysis3.csv',separator=',', append=False)
cnn.fit(X_train, y_train, nb_epoch=10,validation_data=(X_test, y_test),callbacks=[checkpointer,csv_logger])
cnn.save("results/cnn3-LSTMresults/cnn_model.hdf5")

loss, accuracy = cnn.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

y_pred = cnn.predict_classes(X_test)
#np.savetxt('res/expected3.txt', y_test, fmt='%01d')
#np.savetxt('res/predicted3.txt', y_pred, fmt='%01d')

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