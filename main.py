# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 00:21:16 2022

@author: aayus
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import cluster, datasets


df = pd.read_csv("data.csv", names=['x','y'])
print(df.head())

print(df.describe())

plt.figure(figsize=(8,5))
plt.title("Half-moon shaped data", fontsize=18)
plt.grid(True)
plt.scatter(df.x,df.y)
plt.show()

ax = df.plot.hist(bins=12, alpha=0.5)
ax = df.plot.kde(bw_method=0.4)

moons = df.to_numpy()

## KMeans

km=cluster.KMeans(n_clusters=2)



km.fit(moons)


plt.figure(figsize=(8,5))
plt.title("KMeans", fontsize=18)
plt.grid(True)
plt.scatter(moons[:,0],moons[:,1],c=km.labels_)
plt.show()

## Gaussian Mixture Model

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2)
gmm.fit(moons)

plt.figure(figsize=(8,5))
plt.title("Gaussian Mixture", fontsize=18)
plt.grid(True)
plt.scatter(moons[:,0],moons[:,1],c=gmm.predict(moons))
plt.show()



clustering = cluster.SpectralClustering(n_clusters=2, n_neighbors= 7,eigen_solver='arpack', affinity="nearest_neighbors").fit(moons)


## Spectral Clustering

labels = clustering.fit_predict(moons)
labels



plt.figure(figsize=(8,5))
plt.title("Spectral Clustering", fontsize=18)
plt.grid(True)
plt.scatter(moons[:,0],moons[:,1],c=labels)
plt.show()


## Agglomerative


from sklearn.cluster import AgglomerativeClustering
cluster_avg = AgglomerativeClustering(n_clusters=2, affinity='l1', linkage='average')  
cluster_avg.fit_predict(moons)
# plt.figure(figsize=(10, 7)) 
plt.figure(figsize=(8,5))
plt.grid(True)
plt.title("Agglomerative Clustering")
plt.scatter(moons[:,0],moons[:,1], c=cluster_avg.labels_)
plt.show()

## DBSCAN

dbs = cluster.DBSCAN(eps=0.0551)

dbs.fit(moons)

plt.figure(figsize=(8,5))
plt.title("DBSCAN", fontsize=18)
plt.grid(True)
plt.scatter(moons[:,0],moons[:,1],c=dbs.labels_)
plt.show()



pd.DataFrame(dbs.labels_).to_csv("Aayushman_test_data_class_labels.csv", index = False, header = False)
np.savetxt('Cluster_Labels.txt',dbs.labels_)

fg = pd.DataFrame(dbs.labels_)

fg.plot.hist()


fg.plot.kde()


