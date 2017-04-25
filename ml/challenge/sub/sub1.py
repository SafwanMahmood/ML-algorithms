import numpy as np
import pandas as pd
import random
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import VotingClassifier

dataset = pd.read_csv("TrainDataMultiClassClassification.csv")
#dataset = dataset.drop(dataset.columns[0], axis=1)		
# dataset = dataset.loc[random.sample(list(dataset.index),dataset.shape[0])]
X = np.array(dataset[dataset.columns[1:dataset.shape[1]-1]])
y = np.array(dataset[dataset.columns[dataset.shape[1]-1]])
from sklearn.svm import SVC
# clf = 
# clf = AdaBoostClassifier(
#     SVC(kernel='rbf'),
#     n_estimato()rs=600,
#     learning_rate=1.5,
#     algorithm="SAMME")
dataset1 = pd.read_csv("TestDataMultiClass.csv")
#dataset1 =dataset1.drop(dataset1.columns[0], axis=1)
X1 = np.array(dataset1[dataset1.columns[1:dataset1.shape[1]]])
# y_pred = clf.predict(X1)
X11 = np.array(dataset1[dataset1.columns[0]])

from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
# clf1 = OneVsOneClassifier(SVC(kernel='rbf'))
from sklearn import grid_search,svm
# # clf1 = clf.fit(X, y)
# # clf1 = BaggingClassifier(DecisionTreeClassifier(criterion='entropy'),max_samples=0.9, max_features=0.4)
# from sklearn.multiclass import OneVsRestClassifier
# clf2 = OneVsRestClassifier(SVC(kernel='rbf'))
parameters = [{'kernel': ['rbf']}]

svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf = clf.fit(X, y)
# clf = BaggingClassifier(OutputCodeClassifier(SVC(kernel='rbf'),code_size=2, random_state=0),max_samples=0.5, max_features=0.5)
# clf = BaggingClassifier(SVC(kernel='rbf'),max_samples=0.5, max_features=0.5)
# eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
# eclf1 = eclf1.fit(X, y)
# pred1 = eclf1.predict(X1)

# eclf2 = VotingClassifier(estimators=[
#          ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
#          voting='soft')
# eclf2 = eclf2.fit(X, y)
# pred2 = eclf2.predict(X1)

# eclf3 = VotingClassifier(estimators=[
#         ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
#         voting='soft', weights=[2,1,1])
# eclf3 = eclf3.fit(X, y)
clf.fit(X, y)
pred1 = clf.predict(X1)



import csv
from itertools import izip

with open('some26.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(('Id', 'class'))
    writer.writerows(izip(X11, pred1))
