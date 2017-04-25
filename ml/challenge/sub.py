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
# create your Ensemble

dataset = pd.read_csv("TrainDataBinaryClassification.csv")
#dataset = dataset.drop(dataset.columns[0], axis=1)		
from sklearn.cross_validation import train_test_split
# dataset = dataset.loc[random.sample(list(dataset.index),dataset.shape[0])]
X_train = np.array(dataset[dataset.columns[1:dataset.shape[1]-1]])
Y_train = np.array(dataset[dataset.columns[dataset.shape[1]-1]])

# clf = BaggingClassifier(DecisionTreeClassifier(criterion='entropy'),max_samples=0.6, max_features=0.6)
# #
# clf = clf.fit(X_train,Y_train)
#
dataset1 = pd.read_csv("TestDataTwoClass.csv")
X1 = np.array(dataset1[dataset1.columns[0]])
dataset1 =dataset1.drop(dataset1.columns[0], axis=1)
X_test = dataset1
# y_tested = clf.predict(X_test)
# X_train2 = X_train
# Y_train2 = Y_train
# X_test2 = X_test
# names = [ "Decision Tree",
#          "Random Forest", "AdaBoost"]

classifiers = [
    BaggingClassifier(DecisionTreeClassifier(criterion = 'entropy', max_depth = 100),max_samples=1.0, max_features=1.0),
    BaggingClassifier(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),max_samples=1.0, max_features=1.0),
    AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1),
    BaggingClassifier(ExtraTreesClassifier(criterion = 'entropy', max_depth = 100,n_estimators=100),max_samples=1.0, max_features=1.0)]
# clfs = classifiers # [clf1, clf2]
# ens = Ensemble(classifiers = clfs)

# # create your Combiner
# # the rules can be 'majority_vote', 'max', 'min', 'mean' or 'median'
# comb = Combiner(rule='max')

# # now create your ensemble classifier
# ensemble_clf = EnsembleClassifier(ensemble=ens, combiner=comb)
# ensemble_clf = ensemble_clf.fit(X_train,Y_train)
# y_tested = ensemble_clf.predict(X_test)


for i in xrange(1,10):
    clf = BaggingClassifier(DecisionTreeClassifier(criterion = 'entropy', max_depth = i + 100),max_samples=1.0, max_features=1.0)
    clf = clf.fit(X_train, Y_train)
    y_tested1 = clf.predict(X_test)
    for a in range(len(y_tested)):
        y_tested[a] = (y_tested[a] & y_tested1[a])
    clf = BaggingClassifier(ExtraTreesClassifier(criterion = 'entropy', max_depth = i + 100,n_estimators=100+i),max_samples=1.0, max_features=1.0)
    clf = clf.fit(X_train, Y_train)
    y_tested2 = clf.predict(X_test)
    for a in range(len(y_tested)):
         y_tested[a] = (y_tested[a] & y_tested2[a])
    pred1 = np.logical_and(y_tested1,y_tested2)
    y_tested = np.logical_and(y_tested,pred1)     
    clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)
    clf = clf.fit(X_train, Y_train)
    pred11 = clf.predict(X_test)
    y_tested = np.logical_and(pred11,y_tested)
    clf = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)  
    clf = clf.fit(X_train, Y_train)
    pred11 = clf.predict(X_test)
    y_tested = np.logical_and(pred11,y_tested)

            
for clf in classifiers:
        clf = clf.fit(X_train, Y_train)
        pred1 = clf.predict(X_test)
        y_tested = np.logical_and(pred1,y_tested)


# for name in names:
#     y_tested = np.logical_and(pred[name],pred[name+1]) 
y_pred = y_tested.astype(np.int64)
# pred = zip(X1 ,prediction1)
# np.savetxt("some.csv", pred , delimiter="," , fmt='%d' )
# df2n=pd.read_csv("some.csv",header = None)
# y_tested = np.array(dataset1[dataset1.columns[1]]);
# print accuracy_score(y_tested,Y_test)
# print Y_test[1272]

# from sklearn import tree
# clf = tree.DecisionTreeClassifier(criterion="entropy")
# # from sklearn.svm import SVC
# # clf = SVC(kernel='linear')
# # from sklearn.ensemble import GradientBoostingRegressor
# # clf = GradientBoostingRegressor(loss='quantile', alpha =0.5,
# #                                 n_estimators=250, max_depth=4,
# #                                 learning_rate=.1, min_samples_leaf=9,
# #                                 min_samples_split=9)
# # from sklearn.ensemble import RandomForestClassifier

# clf.fit(X, y)
#X1 = np.array(dataset1[dataset1.columns[0]])
# X_test = ch2.transform(dataset1)

# y_pred = clf.predict(X_test)
# print(y_pred)

# f = open("foo1.csv", 'w+')
# writer = csv.writer(f)
# for item in X1,y_pred:
# 	writer.writerow((item,item))
       

import csv
from itertools import izip

with open('some66.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(('Id', 'class'))
    writer.writerows(izip(X1, y_pred))
