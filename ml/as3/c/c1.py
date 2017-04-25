import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as ploty
import time
import math
from numpy.linalg import inv
from sklearn.feature_extraction import DictVectorizer as DVec
from sklearn import preprocessing
dataset = pd.read_csv("traindata.csv", header = None,delimiter=',')


# dataset = dataset.T.to_dict().values()
# vec = DVec(sparse=False)
# vec_x = vec.fit_transform(dataset)
# dataset = vec_x


dataset = pd.DataFrame(pd.get_dummies(dataset))

dataset1 = dataset.sample(frac=0.8)
dataset_test = dataset.loc[~dataset.index.isin(dataset1.index)]
y_test = np.array(dataset_test[dataset_test.columns[dataset_test.shape[1]-1]]) 


dataset1 = np.array(dataset[dataset.columns[0:dataset.shape[1]-2]])
Y_train = np.array(dataset[dataset.columns[dataset.shape[1]-1]])


dataset_test = np.array(dataset_test[dataset_test.columns[0:dataset_test.shape[1]-2]])


scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(dataset1)
X = scaler.transform(dataset1)
X = [[1] + x for x in X]
X =np.array(X)

scaler1 = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(dataset_test)
X_test = scaler.transform(dataset_test)
X_test = [[1] + x for x in X_test]
X_test =np.array(X_test)

#remove the comments for deleting
	

def mylinridgereg(X, Y, lamda):
	X_T=np.transpose(X)
	# X=np.delete(X, 2, axis=1)          
    # X=np.delete(X, 4, axis=1)          

	Id = np.ones((len(X[0]),len(X[0])))
	temp = np.dot(X_T,X)+(Id*lamda)
	invF  = inv(temp)
	weights = np.dot(np.dot(invF,X_T),Y)
	return weights


def mylinridgeregeval(X, weights):
	# X=np.delete(X, 2, axis=1)          
    # X=np.delete(X, 4, axis=1)          

	predict = np.dot(X,weights)
	return predict	

def meansquarederr(T, Tdash):
	arr=[]
	for i in range(len(T)):
		arr.append((T[i]-Tdash[i])**2)
		mean_sq=(sum(arr)/len(T))
		mean_sq=np.sqrt(mean_sq)
	return mean_sq
lam = 50
wts = mylinridgereg(X,Y_train,lam)
maxy= np.amax(wts)
miny = np.amin(wts)
preds = mylinridgeregeval(X,wts)
sq_err = meansquarederr(Y_train,preds)
preds_test = mylinridgeregeval(X_test,wts)
sq_err_test = meansquarederr(y_test,preds_test)
print(sq_err_test)
print(sq_err)

for x in range(len(wts)):
	if wts[x] == maxy:

		print x
		print wts[x]   

for x in range(len(wts)):
	if wts[x] == miny:

		print x
		print wts[x]   		