import numpy as np
import pandas as pd
import random
import time

dataset = pd.read_csv("dataset1.txt", header = None,delim_whitespace=True)

X = np.array(dataset[dataset.columns[0:dataset.shape[1]]])

x = np.array(dataset[dataset.columns[0:dataset.shape[1]-1]])
y =np.array(dataset[dataset.columns[dataset.shape[1]-1]])


from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans

kmeans = KMeans()
kmeans = kmeans.fit(X)


k  = len(kmeans.cluster_centers_)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
from matplotlib import pyplot

for i in range(k):
    ds = X[np.where(labels==i)]
    pyplot.plot(ds[:,0],ds[:,1],'o')
    lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
    pyplot.setp(lines,ms=15.0)
print("The number of clusters are ",k)
pyplot.savefig("plot3.png")