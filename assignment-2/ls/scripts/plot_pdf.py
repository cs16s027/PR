from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import numpy as np
from scipy.stats import multivariate_normal

def plotPDF(mus, covs, plot):
    xmin = int(min([mus[label][0] for label in [0, 1, 2]]))
    xmax = int(max([mus[label][0] for label in [0, 1, 2]]))
    ymin = int(min([mus[label][1] for label in [0, 1, 2]]))
    ymax = int(max([mus[label][1] for label in [0, 1, 2]]))
    '''
    # Generate the underlying x-y grid
    x, y = np.mgrid[int(mu[0]) - 5 : int(mu[0]) + 5:100j, int(mu[1]) - 5 : int(mu[1]) + 5 : 500j]
    # Generate tuples (x, y) : X x Y
    xy = np.column_stack([x.flat, y.flat])
    '''
    # Generate the underlying x-y grid
    x, y = np.mgrid[ymin - 7 : ymax + 7 : 100j, ymin - 7 : ymax + 7 : 100j]
    # Generate tuples (x, y) : X x Y
    xy = np.column_stack([x.flat, y.flat])
    base = np.zeros_like(x)
    # Base plot 
    fig = plt.figure(1)
    ax = fig.gca(projection = '3d')
    ax.plot_surface(x, y, base, rstride = 1, cstride = 1, cmap=cm.coolwarm, linewidth = 0, antialiased=False)

    for label in [0, 1, 2]:
        mu, cov = mus[label], covs[label]
        # Compute the gaussian on this grid with mean = mu, covariance = cov
        z = multivariate_normal.pdf(xy, mean = mu, cov = cov)
        # Reshape the points to fall in line with x,y 
        z = z.reshape(x.shape)
        # Contour placement
        z_min, z_max = np.min(z), np.max(z)
        z_range = z_max - z_min
        contour_location = z_min - 2 * z_range

        # Plot the pdf and the contours
        plt.figure(1)
        ax.set_zlim3d(contour_location, z_max)
        #ax.plot_surface(x, y, z, rstride = 8, cstride = 8, alpha = 0.25)
        # Surface plot
        ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap=cm.coolwarm, linewidth = 0, antialiased=False)
        #ax.plot_surface(x, y, z, rstride = 1, cstride = 1, color = 'blue', linewidth = 0, antialiased=False)
        # Contour plot 
        ax.contour(x, y, z, zdir = 'z', offset = contour_location, cmap = cm.coolwarm)

        # Plot the directions
        '''
        _, eigvec = np.linalg.eig(cov)
        eig_extent = np.max(x)
        direction_1 = mu + eig_extent * eigvec[:, 0]
        direction_2 = mu + eig_extent * eigvec[:, 1]
        plt.plot([mu[0], direction_1[0]], [mu[1], direction_1[1]], zs = contour_location, zdir = 'z', color = 'black')
        plt.plot([mu[0], direction_2[0]], [mu[1], direction_2[1]], zs = contour_location, zdir = 'z', color = 'black')
        '''

    plt.savefig(plot)

if __name__ == '__main__':
    plotPDF(np.array([1.0, 2.0]), np.array([[1, 0.6], [0.6, 1]]))
    plt.show()

