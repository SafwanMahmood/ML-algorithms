import numpy as np
import pandas as pd
import random
import csv
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


dataset = pd.read_csv("TrainDataBinaryClassification.csv")
#dataset = dataset.drop(dataset.columns[0], axis=1)		
from sklearn.cross_validation import train_test_split
# dataset = dataset.loc[random.sample(list(dataset.index),dataset.shape[0])]
X_train = np.array(dataset[dataset.columns[1:dataset.shape[1]-1]])
Y_train = np.array(dataset[dataset.columns[dataset.shape[1]-1]])

clf = BaggingClassifier(DecisionTreeClassifier(criterion='entropy'),max_samples=0.6, max_features=0.6)
clf = clf.fit(X_train,Y_train)

dataset1 = pd.read_csv("TestDataTwoClass.csv")
X1 = np.array(dataset1[dataset1.columns[0]])
dataset1 =dataset1.drop(dataset1.columns[0], axis=1)
X_test = dataset1
y_tested = clf.predict(X_test)

classifiers = [
    BaggingClassifier(DecisionTreeClassifier(criterion = 'entropy', max_depth = 100),max_samples=1.0, max_features=1.0),
    BaggingClassifier(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),max_samples=1.0, max_features=1.0),
    AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1),
    BaggingClassifier(ExtraTreesClassifier(criterion = 'entropy', max_depth = 100,n_estimators=100),max_samples=1.0, max_features=1.0)]

for i in xrange(1,10):
    clf = BaggingClassifier(DecisionTreeClassifier(criterion = 'entropy', max_depth = i + 2),max_samples=1.0, max_features=1.0)
    clf = clf.fit(X_train, Y_train)
    y_tested1 = clf.predict(X_test)
    for a in range(len(y_tested)):
        y_tested[a] = (y_tested[a] & y_tested1[a])
    clf = BaggingClassifier(ExtraTreesClassifier(criterion = 'entropy', max_depth = i + 2,n_estimators=80),max_samples=1.0, max_features=1.0)
    clf = clf.fit(X_train, Y_train)
    y_tested2 = clf.predict(X_test)
    for a in range(len(y_tested)):
         y_tested[a] = (y_tested[a] & y_tested2[a])
    pred1 = np.logical_and(y_tested1,y_tested2)
    y_tested = np.logical_and(y_tested,pred1)     
    
            
for clf in classifiers:
        clf = clf.fit(X_train, Y_train)
        pred1 = clf.predict(X_test)
        y_tested = np.logical_and(pred1,y_tested)


y_pred = y_tested.astype(np.int64)
       

import csv
from itertools import izip

with open('some.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(('Id', 'class'))
    writer.writerows(izip(X1, y_pred))
