import numpy as np
import pandas as pd
import random
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search,svm
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("TrainDataMultiClassClassification.csv")

X = np.array(dataset[dataset.columns[1:dataset.shape[1]-1]])
y = np.array(dataset[dataset.columns[dataset.shape[1]-1]])
from sklearn.svm import SVC
dataset1 = pd.read_csv("TestDataMultiClass.csv")
X1 = np.array(dataset1[dataset1.columns[1:dataset1.shape[1]]])
X11 = np.array(dataset1[dataset1.columns[0]])

from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn import grid_search,svm
# clf = BaggingClassifier(svm.SVC(C=167,gamma=0.0000192))
# clf = clf.fit(X, y)
# y_pred = clf.predict(X1)
clf = BaggingClassifier(OutputCodeClassifier(SVC(kernel='rbf',C=167,gamma=0.0000192),code_size=2, random_state=0))
# clf = BaggingClassifier(SVC(kernel='rbf'),max_samples=0.5, max_features=0.5)
clf = clf.fit(X,y)
y_pred = clf.predict(X1)
pred1 = y_pred


import csv
from itertools import izip

with open('some21.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(('Id', 'class'))
    writer.writerows(izip(X11, pred1))
