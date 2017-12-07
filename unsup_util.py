import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from scipy.sparse import csgraph
from numpy import linalg as LA
import sklearn.cluster as clst
import sklearn
import sys

colors = np.array(['red','cyan','yellow'])

def Run_Kmeans(X, k) : 
    
    lb_km = clst.KMeans(n_clusters=k).fit(X)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(X[:,0], X[:,1], c=colors[lb_km.labels_], s=50)
    ax1.set_title("Kmeans Result")    
    return

def Run_unnorm_SC(simiarity_mtx, k, X) : 
    L = csgraph.laplacian(simiarity_mtx, normed=False)

    w, v = LA.eig(L)
    eig_vals_sorted = np.sort(w)
    eig_vecs_sorted = v[:, w.argsort()]

    z = eig_vecs_sorted[:,0:k]

    lb = clst.KMeans(n_clusters=k).fit(z)

    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)    
    ax0.plot(eig_vals_sorted[0:10], 'o-', markersize=10)   
    ax0.set_title("Smallest Eignvalues")   
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(z[:,0], z[:,1], c=colors[lb.labels_], s=50)
    ax1.set_title("Spectral Embedding Space")   
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(X[:,0], X[:,1], c=colors[lb.labels_], s=50)
    ax2.set_title("Spectral Clustering Result")
    return 

def Load_and_show_data(Name):
    path = "clust_datasets/"+Name+".txt"
    l = np.loadtxt(path, dtype='f', delimiter='\t')
    if Name == 'moon' : 
        k = 2
    else :
        k = 3
   
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(l[:,0], l[:,1], s=50)
    ax1.set_title("Data : {}".format(Name))
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title('Ground-Truth')
    ax2.scatter(l[:,0], l[:,1], c=colors[np.int32(l[:,2]-1)], s=50)
    
    return l[:,0:2], k