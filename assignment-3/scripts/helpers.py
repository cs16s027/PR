import h5py
import numpy as np
from sklearn.cluster import KMeans

def cluster(data, K):
    kmeans = KMeans(n_clusters = K, random_state = 0).fit(data)
    return kmeans.cluster_centers_, kmeans.labels_

def loadData(data_path):
    data_ = h5py.File(data_path, 'r')
    data = {}
    for i in ['0', '1', '2']:
        points = data_[i][:]
        data[int(i)] = points.reshape((-1, 23))
    return data

# Compute the multivariate Gaussian distribution
def multivariateNormalPDF(xy, mean, cov):
    z = []
    det_cov = np.linalg.det(cov)
    factor = 1.0 / np.sqrt(np.power(2 * np.pi, 2) * np.linalg.det(cov))
    for x, y in xy:
        point = np.array([x, y])
        exponent = -0.5 * np.dot(np.dot((point - mean).T, np.linalg.inv(cov)), (point - mean))
        z.append(np.exp(exponent))
    z = np.array(z) * factor
    return z
