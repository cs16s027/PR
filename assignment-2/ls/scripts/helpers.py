import numpy as np

# Compute the multivariate Gaussian distribution
def multivariateNormalPDF(xy, mean, cov):
    z = []
    det_cov = np.linalg.det(cov)
    factor = 1.0 / np.sqrt(np.power(2 * np.pi, 2) * np.linalg.det(cov))
    for x, y in xy:
        point = np.array([x, y])
        exponent = -0.5 * np.dot(np.dot((point - mean).T, np.linalg.inv(cov)), (point - mean))
        z.append(np.exp(exponent))
    reshape_size = int(np.sqrt(xy.shape[0])), int(np.sqrt(xy.shape[0]))
    z = np.array(z).reshape(reshape_size) * factor
    return z

