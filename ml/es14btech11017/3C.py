import numpy as np
import pandas as pd
import random
import math
from sklearn.feature_extraction import DictVectorizer as DVec
import time
from sklearn.feature_selection import SelectKBest, chi2
import sys

file_name = sys.argv[1]
start_time = time.clock()
dataset = pd.read_csv(file_name)

dataset.columns =["Patient_ID","Local_tumor_recurrence","Gender","HPV/p16_status","Age_at_diagnosis","Race","Tumor_side","Tumor_subsite","T_category","N_category","AJCC_Stage","Pathological_grade","Smoking_status_at_diagnosis","Smoking_Pack-Years","Radiation_treatment_course_duration","Total_prescribed_Radiation_treatment_dose","#_Radiation_treatment_fractions","Induction_Chemotherapy","Concurrent_chemotherapy","KM_Overall_survival_censor","none"]
dataset = dataset.loc[random.sample(list(dataset.index),dataset.shape[0])]
dataset['Smoking_Pack-Years'].fillna(0,inplace =2)

dataset = dataset.drop(dataset.columns[0], axis=1)
dataset = dataset.drop(dataset.columns[19], axis=1)	

dataset['Pathological_grade'].fillna('Na',inplace =2)
for i in xrange(0,dataset.shape[1]):
	 if(str(dataset.dtypes[i]) == "object"):
	     dataset =  dataset[dataset[(dataset.columns[i])] != 'Na'] 
y = np.array(dataset[dataset.columns[18]])
X = np.array(dataset[dataset.columns[0:17]])

dataset = dataset.drop(dataset.columns[18], axis=1)
print(dataset.shape)
dataset = dataset.T.to_dict().values()
vec = DVec(sparse=False)
vec_x = vec.fit_transform(dataset)
dataset = vec_x

		
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.10, random_state=42)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)
ch2 = SelectKBest(chi2, k=10)

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,y_train)

from sklearn.externals.six import StringIO  
from IPython.display import Image
import pydotplus
dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data, filled=True) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("datatree.pdf")
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_test)
print("The accuracy is : ")
print(acc)
print("Time in seconds :")
print((time.clock() - start_time))