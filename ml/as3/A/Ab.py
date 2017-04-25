import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as ploty
import time
dataset = pd.read_csv("traindata.csv", header = None,delim_whitespace=True)
from sklearn.cross_validation import train_test_split
X = np.array(dataset[dataset.columns[0:dataset.shape[1]-2]])
y = np.array(dataset[dataset.columns[dataset.shape[1]-1]])
n = dataset.shape[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestClassifier
start_time = time.clock()
oob_error = []
m =[]
err=[]
rec=[]
from sklearn.metrics import recall_score
from sklearn import metrics 
for l in xrange(1,56):
	clf = RandomForestClassifier(n_estimators=80,max_features=l,oob_score=True)
	clf.fit(X_train, y_train) 
	oob_error.append(1 - clf.oob_score_)
	y_pred = clf.predict(X_test)
	err.append((pow(abs(y_pred-y_test),2)))
	m.append(l)
err = np.sum(err, axis=1)
err = np.sqrt(np.abs(err))
myInt = np.sqrt(n)
err = [x / myInt for x in err]
ploty.xlabel('m')
ploty.ylabel('error')
ploty.plot(m,oob_error, color='blue', linestyle='dashed',marker='x')
ploty.plot(m,err, color='red', linestyle='dashed',marker='x')
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_test)
print("The accuracy is : ")
print(acc)
print("Time in seconds :")
print((time.clock() - start_time))
ploty.savefig("plot4.png")
