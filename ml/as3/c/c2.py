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


#dataset = dataset.T.to_dict().values()
#vec = DVec(sparse=False)
#vec_x = vec.fit_transform(dataset)
#dataset = vec_x
dataset = pd.DataFrame(pd.get_dummies(dataset))

print(dataset)
dataset = dataset.loc[random.sample(list(dataset.index),dataset.shape[0])]
dataset1 = dataset.sample(frac=0.6)
dataset_test = dataset.loc[~dataset.index.isin(dataset1.index)]
y_test = np.array(dataset_test[dataset_test.columns[dataset_test.shape[1]-4]]) 


dataset1 = np.array(dataset[dataset.columns[0:dataset.shape[1]-2]])
Y_train = np.array(dataset[dataset.columns[dataset.shape[1]-4]])


dataset_test = np.array(dataset_test[dataset_test.columns[0:dataset_test.shape[1]-2]])


scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(dataset1)
X = scaler.transform(dataset1)
X = [[1] + x for x in X]
X =np.array(X)

scaler1 = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(dataset_test)
X_test = scaler.transform(dataset_test)
X_test = [[1] + x for x in X_test]
X_test =np.array(X_test)


def mylinridgereg(X, Y, lamda):
	X_T=np.transpose(X)
	Id = np.ones((len(X[0]),len(X[0])))
	temp = np.dot(X_T,X)+(Id*lamda)
	invF  = inv(temp)
	weights = np.dot(np.dot(invF,X_T),Y)
	return weights


def mylinridgeregeval(X, weights):
	predict = np.dot(X,weights)
	return predict	

def meansquarederr(T, Tdash):
	arr=[]
	for i in range(len(T)):
		arr.append((T[i]-Tdash[i])**2)
		mean_sq=(sum(arr)/len(T))
		mean_sq=np.sqrt(mean_sq)
	return mean_sq

g_pred =[]
g_pred1 =[]
lamdas = []
lams = [1,5,10,15,20,25,30,35,40,45,50,55,60]
for lam in lams:	
	wts = mylinridgereg(X,Y_train,lam)
	maxy= np.amax(wts)
	preds = mylinridgeregeval(X,wts)
	sq_err = meansquarederr(Y_train,preds)
	g_pred.append(sq_err)
	preds_test = mylinridgeregeval(X_test,wts)
	sq_err_test = meansquarederr(y_test,preds_test)
	g_pred1.append(sq_err_test)
	lamdas.append(lam)
miny= np.amin(g_pred)
miny1= np.amin(g_pred1)

for x in range(len(g_pred1)):
	if g_pred1[x]==miny1:
		print x
print(miny)
print(miny1)

ploty.xlim(1,60)
ploty.xlabel('lamda')
ploty.ylabel('error')
# ploty.plot(lamdas,g_pred, color='blue',marker='x')
ploty.plot(lamdas,g_pred1, color='red',marker='x')
ploty.savefig("plot31.png")



# # print(preds_test)
# print(sq_err_test)
# # print(preds)
# print(sq_err)


# for x in range(len(wts)):
# 	if wts[x] == maxy:

# 		print x
# 		print wts[x]   