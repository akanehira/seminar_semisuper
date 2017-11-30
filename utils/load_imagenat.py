import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn.decomposition import PCA
from utils.load_mnist import extract_mnist_data

def load_data(name='iris'):
    ''' We call this dataset as imagenat dataset '''
    if name == 'imagenat':
        name = 'mnist'
    if name == 'iris':
        iris = datasets.load_iris()
        X = iris.data[:, :2]  # we only take the first two features.
        y = iris.target
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)
     
        return X_train, X_test, y_train, y_test 
    elif name == 'mnist':
        y1, y2 = 3, 8
        n_sample = 500
        X_train, X_test, y_train, y_test = extract_mnist_data()
        X_train = X_train[np.logical_or(y_train==y1, y_train==y2)][:n_sample]
        y_train = y_train[np.logical_or(y_train==y1, y_train==y2)][:n_sample]
        pca = PCA(n_components=2)
        X_train = pca.fit_transform(X_train)
        
        X_test = X_test[np.logical_or(y_test==y1, y_test==y2)][:n_sample]
        y_test = y_test[np.logical_or(y_test==y1, y_test==y2)][:n_sample]
        X_test = pca.transform(X_test)
        
        return X_train, X_test, y_train, y_test 
    else:
        n_sample = 200
        X_train = np.random.multivariate_normal([-1, -1], 0.6*np.eye(2), size=n_sample)
        X_test = np.random.multivariate_normal([1, 1], 0.6*np.eye(2), size=n_sample)
    
        y_train = np.ones(n_sample).astype(np.uint8) 
        y_test = np.ones(n_sample).astype(np.uint8) + 1
     
        return X_train, X_test, y_train, y_test 