import numpy as np
import h5py

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

# Save trained model
def saveModel(params, path):
    print 'Saving model'
    model = h5py.File(path, 'w')
    para_dict = [(0, 'mus'), (1 , 'covs')]
    for index, name in para_dict:
        model.create_dataset(name = name, data = params[index], shape = params[index].shape, dtype = params[index].dtype)
    model.close()

# Load trained model
def loadModel(path):
    print 'Loading model'
    params = ['', '']
    model = h5py.File(path, 'r')
    para_dict = [(0, 'mus'), (1 , 'covs')]
    for index, name in para_dict:
        params[index] = model[name][:]
    return params



