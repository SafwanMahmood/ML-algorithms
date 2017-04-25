
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import random
import time
import matplotlib.cm as cm
import numpy as np


dataset = pd.read_csv("dataset2.txt", header = None,delim_whitespace=True)


X = np.array(dataset[0:dataset.columns[0]-1])
y =np.array(dataset[dataset.columns[dataset.shape[1]-1]])


       
k = 3
clusterer = KMeans(n_clusters=k, random_state=10)
cluster_labels = clusterer.fit_predict(X)
colors = cm.spectral(cluster_labels.astype(float) / k)
plt.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,c=colors)

centers = clusterer.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1],marker='o', c="white", alpha=1, s=200)

for i, c in enumerate(centers):
    plt.scatter(c[0], c[1], marker='$%d$' % (i+1), alpha=1, s=50)

plt.title("The visualization of the clustered data.")
plt.xlabel("X")
plt.ylabel("Y")

plt.suptitle(("KMeans clustering with n_clusters = %d" % k),fontsize=14, fontweight='bold')

plt.savefig("plot2.png")