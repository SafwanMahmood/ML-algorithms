import numpy as np
import pandas as pd
import random
import time

dataset = pd.read_csv("data.csv", header = None)
dataset.columns = ["Sample ID", "Clump Thickness", "Uniformity of Cell Size ", "Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]
for i in xrange(0,dataset.shape[1]):
	 if(str(dataset.dtypes[i]) == "object"):
	     dataset =  dataset[dataset[(dataset.columns[i])] != "?"] 
dataset = dataset.drop(dataset.columns[0], axis=1)		
from sklearn.cross_validation import train_test_split
dataset = dataset.loc[random.sample(list(dataset.index),dataset.shape[0])]
X = np.array(dataset[dataset.columns[0:8]])
y = np.array(dataset[dataset.columns[9]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
start_time = time.clock()
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_test)
print("The accuracy is : ")
print(acc)
print("Time in seconds :")
print((time.clock() - start_time))