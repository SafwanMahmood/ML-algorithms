import os
from PIL import Image
import random 
import pylab
import numpy as np
import math
import cv2
import glob
from sklearn.decomposition import PCA
from numpy import linalg 
from sklearn.preprocessing import scale
image_list = []
import matplotlib . cm as cm
from matplotlib import pyplot as plt


for filename in glob.glob('faces_4/*/'):
    for img in glob.glob((str)(filename)+'*.pgm'):
        im = cv2.imread(img,0)
        shaper = np.array(im).shape
        m,n  = np.array(im).shape[0:2]
        image_list.append(np.array(im).flatten())


flag = 0
flag1 = 0
flag2 =0
scaled_imglist = scale(np.array(image_list).astype(float))



def PCA(X ) :
    X = np.transpose(X)
    no_img ,dim = X.shape
    mean_X = X.mean(axis =0).astype(float)
    num_components =no_img
    X = X - mean_X
    X = np.cov(X)
    eigenvalues , eigenVec = np.linalg.eig(X)
    U, s, V = np.linalg.svd(X)

    eigenVec = np.dot(np.transpose(X),eigenVec)
    eigenVec1 = eigenVec / np.linalg.norm(eigenVec)

    for i in xrange (no_img):
        eigenVec [: , i ] = eigenVec [: , i ]/ np . linalg . norm ( eigenVec [: , i ])
    
    indexs = np.argsort(-eigenvalues)
    eigenVec = eigenVec [:,indexs]
    s = sorted(s,reverse = True) 
    return eigenvalues,s,eigenVec,mean_X


def variance(eig_vals):
    sum1 = sum(eig_vals)
    var_exp = [(i / sum1) for i in sorted(eig_vals, reverse=True)]
    var_exp1 = np.cumsum(var_exp)
    return var_exp, var_exp1

def transform (X , low , high , dtype = None ):
    X = np.asarray(X)
    minX,maxX = np.min(X) , np.max(X)
    X = X-float(minX)
    X = X /float((maxX-minX))
    X = X*(high-low)
    X = X + low
    return np.asarray(X)




eigv, S , eigVec , immean = PCA(np.array(image_list))

var_exp,var_exp1 = variance(eigv) 

S = S/np.max(S)

for j in range(len(var_exp1)):
    if(var_exp1[j]>0.50 and not flag):
        print "For 50% variance we need ",(j+1)," principal components" 
        flag = 1
    if(var_exp1[j]>0.80 and not flag1):
        print "For 80% variance we need " ,(j+1)," principal components" 
        flag1 = 1
    if(var_exp1[j]>0.90 and not flag2):
        print "For 90% variance we need ",(j+1)," principal components"  
        break


eig_new = np.sqrt((eigv)*(959))
eig_new = eig_new/np.max(eig_new)

eig_vec = [[eig_new[i],eigVec[:,i]] for i in range(len(eig_new))]

for i in range(len(S)):
  plt.scatter(i, S[i])
  plt.plot(i, S[i])
plt.show()


eig_vec.sort(key = lambda i:i[0],reverse= True)


img1 = eig_vec[0]

img2 = transform(img1[1],0,255)
print immean.shape
immean = img2.reshape(m,n)
fig = pylab.figure()
fig.add_subplot(2, 2 ,1)
pylab.gray()
pylab.imshow(immean.astype(int))

img1 = (eig_vec[1])

img2 = transform(img1[1],0,255)

immean = img2.reshape(m,n)
fig.add_subplot(2, 2 ,2)
pylab.gray()
pylab.imshow(immean.astype(int))

img1 = (eig_vec[2])

img2 = transform(img1[1],0,255)

immean = img2.reshape(m,n)
fig.add_subplot(2, 2 ,3)
pylab.gray()
pylab.imshow(immean.astype(int))
# pylab.show()

img1 = (eig_vec[3])

img2 = transform(img1[1],0,255)

immean = img2.reshape(m,n)
fig.add_subplot(2, 2 ,4)
pylab.gray()
pylab.imshow(immean.astype(int))
pylab.show()


PCA_comp =[]

for i,j in eig_vec: 
  PCA_comp.append(j)

img_comp = PCA_comp[0]                 
X = (img_comp.T).dot(img_comp)
X_conv = scaled_imglist[0].dot(X)
immean = transform(X_conv,0,255).reshape(m,n)
fig1 = pylab.figure()
fig1.add_subplot(2, 2 ,1)
pylab.gray()
pylab.imshow(immean.astype(int))
#For 2 principal component
img_comp = PCA_comp[:2]                 
X = (np.transpose(img_comp)).dot(img_comp)
X_conv = scaled_imglist[0].dot(X)
immean = transform(X_conv,0,255).reshape(m,n)
fig1.add_subplot(2, 2 ,2)
pylab.gray()
pylab.imshow(immean.astype(int))
# pylab.show()
 #For 10 principal component
img_comp = PCA_comp[:10]                 
X = (np.transpose(img_comp)).dot(img_comp)
X_conv = scaled_imglist[0].dot(X)
immean = transform(X_conv,0,255).reshape(m,n)
fig1.add_subplot(2, 2 ,3)
pylab.gray()
pylab.imshow(immean.astype(int))
# pylab.show()
 #For 100 principal components
img_comp = PCA_comp[:100]               
X = (np.transpose(img_comp)).dot(img_comp)
X_conv = scaled_imglist[0].dot(X)
immean = transform(X_conv,0,255).reshape(m,n)

fig1.add_subplot(2, 2 ,4)
pylab.gray()
pylab.imshow(immean.astype(int))
pylab.show()
exit(0)