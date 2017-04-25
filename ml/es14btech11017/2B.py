from sklearn.metrics import classification_report, accuracy_score
from sklearn import cross_validation
from operator import itemgetter
from sklearn.cross_validation import train_test_split
import numpy as np
import math
from collections import Counter
import pandas as pd
import random
import time

def get_distance(x, y):
    points = zip(x, y)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))
 
def neigbours_set(training_set, test_instance, k):
    distances = [distance_tuple(training_instance, test_instance) for training_instance in training_set]
    sorted_distances = sorted(distances, key=itemgetter(1))
    sorted_training_instances = [tuple[0] for tuple in sorted_distances]
    return sorted_training_instances[:k]
 
def distance_tuple(training_instance, test_instance):
    return (training_instance, get_distance(test_instance, training_instance[0]))
 
def majority(neighbours):
    classes = [neighbour[1] for neighbour in neighbours]
    count = Counter(classes)
    return count.most_common()[0][0] 

dataset = pd.read_csv("data.csv", header = None)
dataset.columns = ["ID", "Clump Thickness", "Uniformity of Cell Size ", "Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]
for i in xrange(0,dataset.shape[1]):
     if(str(dataset.dtypes[i]) == "object"):
         dataset =  dataset[dataset[(dataset.columns[i])] != "?"]
         dataset[dataset.columns[i]] = dataset[dataset.columns[i]].astype(int)
dataset = dataset.drop(dataset.columns[0], axis=1)      

dataset = dataset.loc[random.sample(list(dataset.index),dataset.shape[0])]
X = np.array(dataset[dataset.columns[0:8]])
y = np.array(dataset[dataset.columns[9]]) 

 
 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2, random_state=42)

train = np.array(zip(X_train,y_train))
test = np.array(zip(X_test, y_test))
 
predictions = []
 
k = 5
start_time = time.clock()  
for x in range(len(X_test)):
 
        neighbours = neigbours_set(train, test[x][0], 5)
        majority_vote = majority(neighbours)
        predictions.append(majority_vote)
        
print ("The accuracy is: ")
acc=accuracy_score(y_test, predictions)
print(acc) 
print("Time in seconds: ")
print((time.clock() - start_time))
