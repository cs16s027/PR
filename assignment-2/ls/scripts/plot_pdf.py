from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import numpy as np
from scipy.stats import multivariate_normal

def plotPDF(mu, cov):
    # Generate the underlying x-y grid
    x, y = np.mgrid[-5.0:5.0:100j, -5.0:5.0:100j]
    # Generate tuples (x, y) : X x Y
    xy = np.column_stack([x.flat, y.flat])

    # Compute the gaussian on this grid with mean = mu, covariance = cov
    z = multivariate_normal.pdf(xy, mean = mu, cov = cov)
    # Reshape the points to fall in line with x,y 
    z = z.reshape(x.shape)
    # Contour placement
    z_min, z_max = np.min(z), np.max(z)
    z_range = z_max - z_min
    contour_location = z_min - 2 * z_range

    # Plot the pdf and the contours
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_zlim3d(contour_location, z_max)
    #ax.plot_surface(x, y, z, rstride = 8, cstride = 8, alpha = 0.25)
    # Surface plot
    ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap=cm.coolwarm, linewidth = 0, antialiased=False)
    # Contour plot 
    conts = ax.contour(x, y, z, zdir = 'z', offset = contour_location, cmap = cm.coolwarm)
    print conts


    # Plot the directions
    _, eigvec = np.linalg.eig(cov)
    eig_extent = np.max(x)
    direction_1 = mu + eig_extent * eigvec[:, 0]
    direction_2 = mu + eig_extent * eigvec[:, 1]
    plt.plot([mu[0], direction_1[0]], [mu[1], direction_1[1]], zs = contour_location, zdir = 'z', color = 'black')
    plt.plot([mu[0], direction_2[0]], [mu[1], direction_2[1]], zs = contour_location, zdir = 'z', color = 'black')


if __name__ == '__main__':
    plotPDF(np.array([1.0, 2.0]), np.array([[1, 0.6], [0.6, 1]]))
    plt.show()

