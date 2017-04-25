import numpy as np
import pandas as pd
import random
import math
from sklearn.feature_extraction import DictVectorizer as DVec
import matplotlib.pyplot as ploty
import time
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


        
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.10)

def gain(data, attr, target_attr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    val_freq       = {}
    subset_entropy = 0.0

    for record in data:
        if (val_freq.has_key(record[attr])):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]]  = 1.0
    for val in val_freq.keys():
        val_prob        = val_freq[val] / sum(val_freq.values())
        data_subset     = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    return (entropy(data, target_attr) - subset_entropy)


def entropy(data, target_attr):
    val_freq     = {}
    data_entropy = 0.0

    for record in data:
        if (val_freq.has_key(record[target_attr])):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]]  = 1.0

    for freq in val_freq.values():
        data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
        
    return data_entropy



def create_decision_tree(data, attributes, target_attr, fitness_func):
    data    = data[:]
    vals    = [record[target_attr] for record in data]
    default = majority_value(data, target_attr)

    if not data or (len(attributes) - 1) <= 0:
        return default
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        best = choose_attribute(data, attributes, target_attr,
                                fitness_func)

        tree = {best:{}}

        for val in get_values(data, best):
            subtree = create_decision_tree(
                get_examples(data, best, val),
                [attr for attr in attributes if attr != best],
                target_attr,
                fitness_func)
            tree[best][val] = subtree

    return tree



