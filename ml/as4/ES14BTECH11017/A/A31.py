import math
import numpy as np
import pandas as pd
import random
import time
import matplotlib.cm as cm
dataset = pd.read_csv("dataset1.txt", header = None,delim_whitespace=True)
X = np.array(dataset[dataset.columns[0:dataset.shape[1]]])

x = np.array(dataset[dataset.columns[0]])
y =np.array(dataset[dataset.columns[dataset.shape[1]-1]])



class_bool = False
noise = None

def neighbours(p,q,eps):
    return math.sqrt(np.power(p-q,2).sum()) < eps

def checker(m, point_id, eps):
    n_points = m.shape[1]
    points = []
    for i in range(0, n_points):
        if neighbours(m[:,point_id], m[:,i], eps):
            points.append(i)
    return points

def cluster_add(m, classes, point_id, cluster_id, eps, min_points):
    points = checker(m, point_id, eps)
    if len(points) < min_points:
        classes[point_id] = noise
        return False
    else:
        classes[point_id] = cluster_id
        for seed_id in points:
            classes[seed_id] = cluster_id
            
        while len(points) > 0:
            current_point = points[0]
            results = checker(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    point_res = results[i]
                    if classes[point_res] == class_bool or \
                       classes[point_res] == noise:
                        if classes[point_res] == class_bool:
                            points.append(point_res)
                        classes[point_res] = cluster_id
            points = points[1:]
        return True
        
def DBSCAN(m, eps, min_points):
    
    cluster_id = 1
    n_points = m.shape[1]
    classes = [class_bool] * n_points
    for point_id in range(0, n_points):
        point = m[:,point_id]
        if classes[point_id] == class_bool:
            if cluster_add(m, classes, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classes

a = np.array([np.array(x),np.array(y)])
import matplotlib.pyplot as pyplot
o = DBSCAN(a,0.5,10)
unique_labels = np.array(o)
n_clusters_ = np.max(o)
k = n_clusters_

colors = cm.spectral(unique_labels.astype(float) / k)
pyplot.scatter(X[:, 0], X[:, 1], marker='.', s=40, lw=0, alpha=0.7,c=colors)
pyplot.title("The visualization of the clustered data.")
pyplot.xlabel("X")
pyplot.ylabel("Y")

pyplot.suptitle(("DBSCAN clustering with n_clusters = %d" % k),fontsize=14, fontweight='bold')

print("The number of clusters are ",k)
pyplot.savefig("plot22.png")
