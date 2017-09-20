from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from helpers import multivariateNormalPDF

import numpy as np

def plotContours(mus, covs, plot, title, figure = 1):
    print 'Plotting contours'
    # Initialize the matplotlib figure object
    fig = plt.figure(figure)
    # A 3d axes object
    ax = fig.gca()

    # Compute the gaussian for each class
    xs, ys, zs = [], [], []
    for label in [0, 1, 2]:
        mu, cov = mus[label], covs[label]
        x, y = np.mgrid[mu[0] - 5 : mu[0] + 5 : 100j, mu[1] - 5 : mu[1] + 5 : 100j]
        xy = np.column_stack([x.flat, y.flat])
        # Compute the gaussian on this grid with mean = mu, covariance = cov
        z = multivariateNormalPDF(xy, mu, cov)
        z = z.reshape(x.shape)
        # Populate Gaussians : zs
        xs.append(x)
        ys.append(y)
        zs.append(z)
    
    # Plot the contours
    for label in [0, 1, 2]:
        # Contour plot 
        ax.contour(xs[label], ys[label], zs[label], cmap = cm.coolwarm)

        # Plot the directions
        mu, cov = mus[label], covs[label]
        _, eigvec = np.linalg.eig(cov)
        eig_extent = 1.5
        direction_1 = mu + eig_extent * eigvec[:, 0]
        direction_2 = mu + eig_extent * eigvec[:, 1]
        plt.plot([mu[0], direction_1[0]], [mu[1], direction_1[1]], color = 'black')
        plt.plot([mu[0], direction_2[0]], [mu[1], direction_2[1]], color = 'black')

    # Label the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    # Save the figure
    plt.legend()
    plt.title(title)
    plt.savefig(plot)

if __name__ == '__main__':
    plotContours([np.array([1.0, 2.0]),\
             np.array([5.0, 7.0]),\
             np.array([-1.0, 11.0])],\
                
            [np.array([[1, 0.6], [0.6, 1]]),\
             np.array([[1.5, 0.3], [0.3, 1.5]]),\
             np.array([[1.2, 0], [0, 1.2]])],\
            'Test plot',\
            'plots/contours.jpg')
    plt.show()

