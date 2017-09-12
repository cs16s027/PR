from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.stats import multivariate_normal

def plotPDF(mu, covariance, color):
    x, y = np.mgrid[-10.0:10.0:500j, -10.0:10.0:500j]
    xy = np.column_stack([x.flat, y.flat])

    z = multivariate_normal.pdf(xy, mean = mu, cov = covariance)
    z = z.reshape(x.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(x, y, z)

if __name__ == '__main__':
    plotPDF(np.array([0.0, 0.0]), np.diag(np.array([1.0, 1.0])), color = 'blue')
    plt.show()
