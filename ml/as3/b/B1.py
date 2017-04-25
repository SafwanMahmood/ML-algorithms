import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as ploty
import time

import numpy as np
from sklearn.naive_bayes import GaussianNB

word=[]
word_test=[]

lines = [line.rstrip('\n') for line in open("traindata.txt","r")]
label = [line.rstrip('\n') for line in open("trainlabels.txt","r")]
lines_test = [line.rstrip('\n') for line in open("testdata.txt","r")]
test_label = [line.rstrip('\n') for line in open("testlabels.txt","r")]
stopword = [line.rstrip('\n') for line in open("stoplist.txt","r")]
#####################################################################
for i in range(0,(len(lines))):
	lines_sep=lines[i].split(" ")
	word.append(lines_sep)
line_words = word
word = [y for x in word for y in x]
# print(len(word))
word = sorted(list(set(word)-set(stopword)))
# print(len(word))
#print (word)
# print(word)

row=[]
X_train=[]
X_train_temp=[]
X_test=[]
X_test_temp=[]
for i in range(0,len(line_words)):
	row=[0]*(len(word)+1)
	row_temp = [0]*(len(word))
	for j in range(len(line_words[i])):
		if(line_words[i][j] in word):
			k=word.index(line_words[i][j])
			row_temp[k]=1
			row[k]=1
	row[len(word)]=int(label[i])
	X_train.append(row_temp)
	#print(len(row))
	X_train_temp.append(row)
with open("preprocess1.txt","w+") as f:
    f.write(",".join((map(str, word))))
    f.write("\n".join(",".join(map(str, x)) for x in X_train_temp))  
################################################################
for i in range(0,(len(lines_test))):
	lines_sep_test=lines[i].split(" ")
	word_test.append(lines_sep_test)
#print(len(word_test))
line_words_test=word_test
for i in range(0,len(line_words_test)):
	row=[0]*(len(word)+1)
	row_temp = [0]*(len(word))
	for j in range(len(line_words_test[i])):
		if(line_words_test[i][j] in word):
			k=word.index(line_words_test[i][j])
			row_temp[k]=1
			row[k]=1
	row[len(word)]=(test_label[i])
	X_test.append(row_temp)
	#print(len(row))
	X_test_temp.append(row)
	

###############################################################

clf = GaussianNB()
clf.fit(X_train,label)

X_test= np.array(X_test)
predicts=[]
counter=0;
# print(test_label[0])
z=clf.predict(X_test)
for i in range(len(X_test)):
	if(test_label[i]==z[i]):
		counter+=1
avg=counter/len(X_test)
# print(avg)
z=[]
z.append(clf.score(X_test, test_label, sample_weight=None))
print(z)
with open("output1.txt","w+") as l:
    l.write(",".join((map(str, z))))
