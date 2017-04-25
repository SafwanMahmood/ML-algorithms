import numpy as np
import pandas as pd
import random
import math
from sklearn.feature_extraction import DictVectorizer as DVec
import matplotlib.pyplot as ploty
import time
import sys

#file_name = sys.argv[1]
start_time = time.clock()
dataset = pd.read_csv("Training.csv.xls")
dataset.columns =["Patient_ID","Local_tumor_recurrence","Gender","HPV/p16_status","Age_at_diagnosis","Race","Tumor_side","Tumor_subsite","T_category","N_category","AJCC_Stage","Pathological_grade","Smoking_status_at_diagnosis","Smoking_Pack-Years","Radiation_treatment_course_duration","Total_prescribed_Radiation_treatment_dose","#_Radiation_treatment_fractions","Induction_Chemotherapy","Concurrent_chemotherapy","KM_Overall_survival_censor","none"]
dataset = dataset.loc[random.sample(list(dataset.index),dataset.shape[0])]

dataset['Smoking_Pack-Years'].fillna(0,inplace =2)
avg = dataset['Smoking_Pack-Years'].mean()
dataset['Smoking_Pack-Years'].fillna(avg,inplace =2)
dataset = dataset.drop(dataset.columns[0], axis=1)
dataset = dataset.drop(dataset.columns[19], axis=1)	

dataset['Pathological_grade'].fillna('Na',inplace =2)
for i in xrange(0,dataset.shape[1]):
	 if(str(dataset.dtypes[i]) == "object"):
	     dataset =  dataset[dataset[(dataset.columns[i])] != 'Na'] 
y = np.array(dataset[dataset.columns[18]])
X = np.array(dataset[dataset.columns[0:17]])

dataset = dataset.drop(dataset.columns[18], axis=1)
dataset = dataset.T.to_dict().values()
vec = DVec(sparse=False)
vec_x = vec.fit_transform(dataset)
dataset = vec_x

		
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.10)
from sklearn import tree
from sklearn.metrics import accuracy_score

x_plot=[]
y_plot=[]
x1_plot=[]
y1_plot=[]


for x in range(25):	
    clf = tree.DecisionTreeClassifier(criterion="entropy",max_leaf_nodes = x+2 )
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_pred1 = clf.predict(X_train)
    acc = accuracy_score(y_pred, y_test)
    acc1 = accuracy_score(y_pred1, y_train)
    ploty.xlabel('No of leaves')
    ploty.ylabel('Accuracy')
    x_plot.append(clf.max_leaf_nodes)
    y_plot.append(acc)
    x1_plot.append(clf.max_leaf_nodes)
    y1_plot.append(acc1)
avg1 = np.mean(y_plot)
print("The avg accuracy is : ")
print(avg1)
ploty.xlim(0,25)
ploty.ylim(0,1)
ploty.plot(x_plot,y_plot, color='blue', linestyle='dashed',marker='x')
ploty.plot(x1_plot,y1_plot, color='red', linestyle='dashed',marker='x')
ploty.savefig("plot31.png")
print("Time in seconds :")
print((time.clock() - start_time))
###############################################
