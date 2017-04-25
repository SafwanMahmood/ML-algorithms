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


dataset1 = np.array(dataset[dataset.columns[0:dataset.shape[1]-2]])
Y = np.array(dataset[dataset.columns[dataset.shape[1]-1]])

scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(dataset1)
X = scaler.transform(dataset1)
X = [[1] + x for x in X]
X =np.array(X)

         
	

def mylinridgereg(X, Y, lamda):
	# X=np.delete(X, 2, axis=1)          
	# X=np.delete(X, 6, axis=1)
	# X=np.delete(X, 4, axis=1) 
	X_T=np.transpose(X)
	Id = np.ones((len(X[0]),len(X[0])))
	temp = np.dot(X_T,X)+(Id*lamda)
	invF  = inv(temp)
	weights = np.dot(np.dot(invF,X_T),Y)
	return weights


def mylinridgeregeval(X, weights):
	# X=np.delete(X, 2, axis=1)          
	# X=np.delete(X, 6, axis=1)
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

wts = mylinridgereg(X,Y,1000)
preds = mylinridgeregeval(X,wts)
sq_err = meansquarederr(Y,preds)
print(wts)
print(preds)
print(sq_err) 