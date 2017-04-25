import numpy as np
import pandas as pd
import random
import time

dataset = pd.read_csv("TrainDataBinaryClassification.csv")
#dataset.columns = ["Sample ID", "Clump Thickness", "Uniformity of Cell Size ", "Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]
# for i in xrange(0,dataset.shape[1]):
# 	 if(str(dataset.dtypes[i]) == "object"):
# 	     dataset =  dataset[dataset[(dataset.columns[i])] != "?"] 
dataset = dataset.drop(dataset.columns[0], axis=1)
print(dataset.shape)		
from sklearn.cross_validation import train_test_split
dataset = dataset.loc[random.sample(list(dataset.index),dataset.shape[0])]
X = np.array(dataset[dataset.columns[0:dataset.shape[1]-1]])
y = np.array(dataset[dataset.columns[dataset.shape[1]-1]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
start_time = time.clock()
# print(y)
# print(X)
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=3,weights='distance')
# from sklearn.svm import SVC
# clf = SVC(kernel='linear')
# from sklearn.ensemble import GradientBoostingRegressor
# clf = GradientBoostingRegressor(loss='quantile', alpha =0.5,
#                                 n_estimators=250, max_depth=4,
#                                 learning_rate=.1, min_samples_leaf=9,
#                                 min_samples_split=9)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
clf= RandomForestClassifier(max_depth = 4)
ch2 = SelectKBest(chi2, k=10)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)

#clf.fit(X, y)
#dataset1 = pd.read_csv("TestDataTwoClass.csv")
#X1 = np.array(dataset1[dataset1.columns[0]])
#dataset1 =dataset1.drop(dataset1.columns[0], axis=1)
#X1 = np.array(dataset1[dataset1.columns[0]])

# from sklearn.ensemble import RandomForestClassifier
# clf= RandomForestClassifier(max_depth = 4)
clf.fit(X_train, y_train)
#X1 = np.array(dataset1[dataset1.columns[0:dataset1.shape[1]]])
# dataset1 = pd.read_csv("TestDataTwoClass.csv")
# dataset1 =dataset1.drop(dataset1.columns[0], axis=1)
# y_pred = clf.predict(X1)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_test)
print("The accuracy is : ")
print(acc)
print("Time in seconds :")
print((time.clock() - start_time))