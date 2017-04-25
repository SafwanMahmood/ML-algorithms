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
clf = RandomForestClassifier(n_estimators=80,max_features=8)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_test)
print("The accuracy is : ")
print(acc)
print("Time in seconds :")
print((time.clock() - start_time))

