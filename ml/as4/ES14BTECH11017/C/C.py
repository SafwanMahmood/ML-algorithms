import numpy as np 
import pandas as pd
import math

from nltk import word_tokenize
from nltk.corpus import reuters 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import re as reg
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

stopwords_list = stopwords.words("english")
def tokenize(text):
    min1 = 3
    words = map(lambda word: word.lower(), word_tokenize(text));
    words = [word for word in words if word not in stopwords_list]
    tokens =(list(map(lambda token: PorterStemmer().stem(token), words)));
    p = reg.compile('[a-zA-Z]+');
    return_vals = list(filter(lambda token: p.match(token) and len(token)>=min1, tokens));
    return return_vals

def tf_compute(docs):   
    clf = TfidfVectorizer(tokenizer = tokenize,min_df=3, max_df=0.90, max_features=1500, use_idf=True, sublinear_tf=True);
    clf = clf.fit_transform(docs);
    return clf.toarray();

UNCLASSIFIED = False
NOISE = None

def _eps_neighborhood(p,q,eps):
    return math.sqrt(np.power(p-q,2).sum()) < eps

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
    
    cluster_id = 1
    n_points = m.shape[1]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        point = m[:,point_id]
        if classifications[point_id] == UNCLASSIFIED:
            if _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications

def findEps(points):
    n = points.shape[0]
    randomVal = random.randrange(0, n)
    four_dist  = []
    for i in range(0,n):
        if not i == randomVal:
            four_dist.append(np.linalg.norm(np.array(points[randomVal])-np.array(points[i])))
    sorted(four_dist)
    return four_dist[3]

topics_labels = ['acq', 'corn', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade', 'wheat']
words_arr = []

counter = []

l = 0;

for x in topics_labels:
    documents = reuters.fileids(x)
    l = l%10;
    for document in documents:
        words_arr.append(reuters.raw(document))
        counter.append(l)
    l = l+1

counter =  np.array(counter)    
X = tf_compute(words_arr)
kmeans = KMeans(n_clusters = 10).fit(X)
labels = kmeans.labels_
count = 0
for i in range(len(labels)):
    if labels[i] == counter[i]:
        count++
recall = recall_score(kmeans.labels_,counter,average = 'weighted')
precision = precision_score(kmeans.labels_,counter,average = 'weighted')
F1 = 2*(precision*recall)/(precision+recall)
RI = adjusted_rand_score(labels,counter)
NMI = normalized_mutual_info_score(labels,counter)
P = count*1.00/len(labels)
print "F1 score,RI score,Normalized Mutual Information scores,purity are ",F1, RI, NMI,P

Val_eps = findEps(X)

values = dbscan(X,Val_eps,4)

n_clusters_dbscan = len(list(set(values)))
print "No of clusters in dbscan",n_clusters_dbscan