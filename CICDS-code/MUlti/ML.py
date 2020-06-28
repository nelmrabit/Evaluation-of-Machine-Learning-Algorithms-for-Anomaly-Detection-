import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, mean_squared_error,
                             mean_absolute_error, roc_curve, classification_report, auc)

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

traindata = np.array(trainX)
trainlabel = np.array(Ytrain)

testdata = np.array(testT)
testlabel = np.array(Ytest)



model = LogisticRegression()
model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
predicted = model.predict(testdata)
np.savetxt('res/predictedLR.txt', predicted, fmt='%01d')

accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="weighted")
precision = precision_score(expected, predicted, average="weighted")
f1 = f1_score(expected, predicted, average="weighted")
cm = metrics.confusion_matrix(expected, predicted)
print(cm)
tpr = float(cm[0][0]) / np.sum(cm[0])
fpr = float(cm[1][1]) / np.sum(cm[1])
print("%.3f" % tpr)
print("%.3f" % fpr)
print("Accuracy")
print("%.3f" % accuracy)
print("precision")
print("%.3f" % precision)
print("recall")
print("%.3f" % recall)
print("f-score")
print("%.3f" % f1)
print("fpr")
print("%.3f" % fpr)
print("tpr")
print("%.3f" % tpr)

print("***************************************************************")


# 2fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
#np.savetxt('res/expectedNB.txt', expected, fmt='%01d')
np.savetxt('res/predictedNB.txt', predicted, fmt='%01d')

accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="weighted")
precision = precision_score(expected, predicted, average="weighted")
f1 = f1_score(expected, predicted, average="weighted")
cm = metrics.confusion_matrix(expected, predicted)
print(cm)
tpr = float(cm[0][0]) / np.sum(cm[0])
fpr = float(cm[1][1]) / np.sum(cm[1])
print("%.3f" % tpr)
print("%.3f" % fpr)
print("Accuracy")
print("%.3f" % accuracy)
print("precision")
print("%.3f" % precision)
print("recall")
print("%.3f" % recall)
print("f-score")
print("%.3f" % f1)
print("fpr")
print("%.3f" % fpr)
print("tpr")
print("%.3f" % tpr)
print("***************************************************************")



# 3 fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
np.savetxt('res/predictedKNN.txt', predicted, fmt='%01d')

# summarize the fit of the model
accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="weighted")
precision = precision_score(expected, predicted, average="weighted")
f1 = f1_score(expected, predicted, average="weighted")

cm = metrics.confusion_matrix(expected, predicted)
print(cm)
tpr = float(cm[0][0]) / np.sum(cm[0])
fpr = float(cm[1][1]) / np.sum(cm[1])
print("%.3f" % tpr)
print("%.3f" % fpr)
print("Accuracy")
print("%.3f" % accuracy)
print("precision")
print("%.3f" % precision)
print("recall")
print("%.3f" % recall)
print("f-score")
print("%.3f" % f1)
print("fpr")
print("%.3f" % fpr)
print("tpr")
print("%.3f" % tpr)
print("***************************************************************")



# 4 fit a DecisionTree model to the data
model = DecisionTreeClassifier()
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
np.savetxt('res/predictedDT.txt', predicted, fmt='%01d')
# summarize the fit of the model
accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="weighted")
precision = precision_score(expected, predicted, average="weighted")
f1 = f1_score(expected, predicted, average="weighted")

cm = metrics.confusion_matrix(expected, predicted)
print(cm)
tpr = float(cm[0][0]) / np.sum(cm[0])
fpr = float(cm[1][1]) / np.sum(cm[1])
print("%.3f" % tpr)
print("%.3f" % fpr)
print("Accuracy")
print("%.3f" % accuracy)
print("precision")
print("%.3f" % precision)
print("recall")
print("%.3f" % recall)
print("f-score")
print("%.3f" % f1)
print("fpr")
print("%.3f" % fpr)
print("tpr")
print("%.3f" % tpr)
print("***************************************************************")

# 5 fit a DAdaBoost model to the data
model = AdaBoostClassifier(n_estimators=10)
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
np.savetxt('res/predictedABoost.txt', predicted, fmt='%01d')
# summarize the fit of the model
accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="weighted")
precision = precision_score(expected, predicted, average="weighted")
f1 = f1_score(expected, predicted, average="weighted")

cm = metrics.confusion_matrix(expected, predicted)
print(cm)
tpr = float(cm[0][0]) / np.sum(cm[0])
fpr = float(cm[1][1]) / np.sum(cm[1])
print("%.3f" % tpr)
print("%.3f" % fpr)
print("Accuracy")
print("%.3f" % accuracy)
print("precision")
print("%.3f" % precision)
print("recall")
print("%.3f" % recall)
print("f-score")
print("%.3f" % f1)
print("fpr")
print("%.3f" % fpr)
print("tpr")
print("%.3f" % tpr)
print("***************************************************************")


# 6 fit a RandomForestClassifier model to the data
model = RandomForestClassifier(n_estimators=10)
model = model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
np.savetxt('res/predictedRF.txt', predicted, fmt='%01d')
# summarize the fit of the model
accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="weighted")
precision = precision_score(expected, predicted, average="weighted")
f1 = f1_score(expected, predicted, average="weighted")

cm = metrics.confusion_matrix(expected, predicted)
print(cm)
tpr = float(cm[0][0]) / np.sum(cm[0])
fpr = float(cm[1][1]) / np.sum(cm[1])
print("%.3f" % tpr)
print("%.3f" % fpr)
print("Accuracy")
print("%.3f" % accuracy)
print("precision")
print("%.3f" % precision)
print("recall")
print("%.3f" % recall)
print("f-score")
print("%.3f" % f1)
print("fpr")
print("%.3f" % fpr)
print("tpr")
print("%.3f" % tpr)
print("***************************************************************")

