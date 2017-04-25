import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as ploty
import time
import numpy as np
from sklearn.naive_bayes import GaussianNB


with open('traindata.txt',"r") as inf:
	data_lines = [line.rstrip('\n') for line in inf]
with open('trainlabels.txt',"r") as inf1:
	data_label = [line.rstrip('\n') for line in inf1]    
with open('testdata.txt',"r") as inf2:
	test_lines = [line.rstrip('\n') for line in inf2]
with open('testlabels.txt',"r") as inf3:
	label_test = [line.rstrip('\n') for line in inf3]	
with open('stoplist.txt',"r") as inf4:
	stoplistwords = [line.rstrip('\n') for line in inf4]

vocab= []
test_voc= []

for i in range(0,(len(data_lines))):
	lines_sep=data_lines[i].split(" ")
	vocab.append(lines_sep)

line_words = vocab

vocab = [i for j in vocab for i in j]

vocab = sorted(list(set(vocab)-set(stoplistwords)))
 
temp=[]
X_train=[]
X_train1=[]
X_test=[]
X_TEST1=[]

vocab.pop(0)

for i in range(0,len(line_words)):
	temp = [0]*(len(vocab)+1)
	temp1 = [0]*(len(vocab))
	for j in range(len(line_words[i])):
		if(line_words[i][j] in vocab):
			k=vocab.index(line_words[i][j])
			temp1[k]=1
			temp[k]=1
	temp[len(vocab)]=(data_label[i])
	X_train.append(temp1)
	X_train1.append(temp)

with open("preprocess.txt","w+") as f:
    f.write(",".join((map(str, vocab))))
    f.write("\n".join(",".join(map(str, x)) for x in X_train1))  

for i in range(0,(len(test_lines))):
	lines_sep_test=data_lines[i].split(" ")
	test_voc.append(lines_sep_test)

line_words_test=test_voc

for i in range(0,len(line_words_test)):
	temp=[0]*(len(vocab)+1)
	temp1 = [0]*(len(vocab))
	for j in range(len(line_words_test[i])):
		if(line_words_test[i][j] in vocab):
			k=vocab.index(line_words_test[i][j])
			temp1[k]=1
			temp[k]=1
	temp[len(vocab)]=(label_test[i])
	X_test.append(temp1)
	X_TEST1.append(temp)
	
X_test= np.array(X_test)

clf = GaussianNB()
clf.fit(X_train,data_label)
pred=clf.predict(X_test)
acc=[]

acc.append(clf.score(X_test, label_test))
print(acc)
with open("output.txt","w+") as l:
    l.write(",".join((map(str, acc))))