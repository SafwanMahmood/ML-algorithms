import numpy as np
import pandas as pd
import random
import time

def classvote(final_res):
	final_arr = np.array(final_res)
	final=[]
	for i in range(len(final_res[0])):
		temp=final_arr[:][:,i]
		counts = np.bincount(temp)
		final.append(np.argmax(counts))
	return final

dataset = pd.read_csv("traindata.csv", header = None,delim_whitespace=True)
dataset_t = dataset.sample(frac=0.7)
dataset_test = dataset.loc[~dataset.index.isin(dataset_t.index)]

y_test = np.array(dataset_test[dataset_test.columns[dataset_test.shape[1]-1]]) 

from sklearn.cross_validation import train_test_split
y_pred1 = []
m =8
start_time = time.clock()
for l in xrange(1,80):
	dataset_t = dataset_t.loc[random.sample(list(dataset_t.index),dataset_t.shape[0])]
	i = random.randint(0,dataset_t.shape[1]-2-m)
	X = np.array(dataset_t[dataset_t.columns[i:i+m]])
	X_test = np.array(dataset_test[dataset_test.columns[i:i+m]])
	y = np.array(dataset_t[dataset_t.columns[dataset_t.shape[1]-1]])
	from sklearn import tree
	clf = tree.DecisionTreeClassifier(criterion='entropy')
	clf.fit(X, y) 
	y_pred1.append(clf.predict(X_test))

y_pred = classvote(y_pred1)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_test)
print("The accuracy is : ")
print(acc)
print("Time in seconds :")
print((time.clock() - start_time))