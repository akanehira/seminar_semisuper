import numpy as np
from sklearn.cluster import KMeans

def get_unlabeled(X, y):
    unlabeled_ind = np.ones(y.shape[0])
    uni = np.unique(y)
    for u in uni:
        ind = np.where(y==u)[0]
        kmeans = KMeans(n_clusters=5)
        cluster = kmeans.fit_predict(X[ind])
        unlabeled_ind[ind[cluster == 3]] = 0
    return unlabeled_ind