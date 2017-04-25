import numpy as np
import pandas as pd
import csv
# from sklearn.neighbors import DescisionTreeClassifier
from sklearn import cross_validation
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
data = pd.read_csv("TrainDataMultiClassClassification.csv")
data2 = pd.read_csv("TestDataMultiClass.csv")
from sklearn import svm
from sklearn.svm import SVC

X_TEST = data2[data2.columns[1:data2.shape[1]-1]]
X1 = data2[data2.columns[0]]



X = data[data.columns[1:data.shape[1]-2]]
Y = data[data.columns[data.shape[1]-1]]
# X_new = SelectKBest(chi2, k=100).fit_transform(X,Y)
from sklearn.ensemble import BaggingClassifier
# X_t,X_TEST,Y_t,Y_TEST = cross_validation.train_test_split(X,Y,test_size = 0.2,random_state = 0)
# clf = OneVsRestClassifier(SVC(kernel='rbf'))
clf = BaggingClassifier(OutputCodeClassifier(SVC(kernel='rbf'),code_size=2, random_state=0))
# clf = BaggingClassifier(SVC(kernel='rbf'))
# clf = svm.SVC(C=130,gamma=0.00001)
clf = clf.fit(X,Y)
pred1 = clf.predict(X_TEST)

# acc = accuracy_score(pred1,Y_TEST)
# print(acc)
# Y_tested = svm.SVC(C=180,gamma=0.0000195).fit(X_train, Y_train).predict(X_test)
# f = open("final.csv", 'w+')
# # writer = csv.writer(f)
# # for item in X1,pred1:
# # 	writer.writerow((item,item))

# prediction1 = pred1.astype(np.int64)
# pred = zip(X_TEST[X_TEST.columns[0]] ,prediction1)
# np.savetxt("some.csv", pred , delimiter="," )

import csv
from itertools import izip

with open('final21.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(('Id', 'class'))
    writer.writerows(zip(X1, pred1))
