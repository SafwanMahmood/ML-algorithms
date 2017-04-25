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



UNCLASSIFIED = False
NOISE = None

def _dist(p,q):
	return math.sqrt(np.power(p-q,2).sum())

def _eps_neighborhood(p,q,eps):
	return _dist(p,q) < eps

def _region_query(m, point_id, eps):
    n_points = m.shape[1]
    seeds = []
    for i in range(0, n_points):
        if _eps_neighborhood(m[:,point_id], m[:,i], eps):
            seeds.append(i)
    return seeds

def _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
    seeds = _region_query(m, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
            
        while len(seeds) > 0:
            current_point = seeds[0]
            results = _region_query(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                       classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True
        
def dbscan(m, eps, min_points):
    """Implementation of Density Based Spatial Clustering of Applications with Noise
    See https://en.wikipedia.org/wiki/DBSCAN
    
    scikit-learn probably has a better implementation
    
    Uses Euclidean Distance as the measure
    
    Inputs:
    m - A matrix whose columns are feature vectors
    eps - Maximum distance two points can be to be regionally related
    min_points - The minimum number of points to make a cluster
    
    Outputs:
    An array with either a cluster id number or dbscan.NOISE (None) for each
    column vector in m.
    """
    cluster_id = 1
    n_points = m.shape[1]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        point = m[:,point_id]
        if classifications[point_id] == UNCLASSIFIED:
            if _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications

a = np.array([np.array(x),np.array(y)])
import matplotlib.pyplot as pyplot
o = dbscan(a,0.5,4)
# print(o)
unique_labels = np.array(o)
n_clusters_ = np.max(o)
# print(n_clusters_)
# core_samples_mask = np.zeros_like(o, dtype=bool)
k = n_clusters_

colors = cm.spectral(unique_labels.astype(float) / k)
pyplot.scatter(X[:, 0], X[:, 1], marker='.', s=40, lw=0, alpha=0.7,c=colors)
pyplot.title("The visualization of the clustered data.")
pyplot.xlabel("X")
pyplot.ylabel("Y")

pyplot.suptitle(("DBSCAN clustering with n_clusters = %d" % k),fontsize=14, fontweight='bold')

print("The number of clusters are ",k)
pyplot.savefig("plot11.png")