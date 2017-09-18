import numpy as np
import h5py

def estimateParams(data, choice):
    means, covs = [], []
    for label in ['0', '1', '2']:
        points = data[label][:]
        mu = np.mean(points, axis = 0)
        if choice[0] == int(label):
            cov = np.dot((points - mu).T, points - mu) / points.shape[0]
            diag = cov[choice[1], choice[1]]
            cov = np.diag([diag, diag])

        means.append(mu)
    covs = [cov, cov, cov] 
    return means, covs 

if __name__ == '__main__':
    data = h5py.File('data/train.h5', 'r')
    means, covs = estimateParams(data)

    np.set_printoptions(precision=3)
    for label in [0, 1, 2]:
        print '###'
        print 'Mean of class -', label
        print means[label]
        print 'Covariance matrix of class -', label
        print covs[label]

