from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import numpy as np

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

def plotPDF(mus, covs, plot):
    # Initialize the matplotlib figure object
    fig = plt.figure(1)
    # A 3d axes object
    ax = fig.gca(projection = '3d')

    # Compute the gaussian for each class
    xs, ys, zs = [], [], []
    for label in [0, 1, 2]:
        mu, cov = mus[label], covs[label]
        x, y = np.mgrid[mu[0] - 5 : mu[0] + 5 : 100j, mu[1] - 5 : mu[1] + 5 : 100j]
        xy = np.column_stack([x.flat, y.flat])
        # Compute the gaussian on this grid with mean = mu, covariance = cov
        z = multivariateNormalPDF(xy, mu, cov)
        # Populate Gaussians : zs
        xs.append(x)
        ys.append(y)
        zs.append(z)
    
    # This computes the location of the contours
    zmin, zmax = np.min(np.array(zs)), np.max(np.array(zs))
    contour_location = zmin - 2 * (zmax - zmin)
    ax.set_zlim3d(contour_location, zmax)

    # Plot the pdf and the contours
    for label in [0, 1, 2]:
        # Surface plot
        ax.plot_surface(xs[label], ys[label], zs[label], rstride = 1, cstride = 1, cmap=cm.coolwarm, linewidth = 0, antialiased=False)
        # Contour plot 
        ax.contour(xs[label], ys[label], zs[label], zdir = 'z', offset = contour_location, cmap = cm.coolwarm)

        # Plot the directions
        mu, cov = mus[label], covs[label]
        _, eigvec = np.linalg.eig(cov)
        eig_extent = np.max(x)
        direction_1 = mu + eig_extent * eigvec[:, 0]
        direction_2 = mu + eig_extent * eigvec[:, 1]
        plt.plot([mu[0], direction_1[0]], [mu[1], direction_1[1]], zs = contour_location, zdir = 'z', color = 'black')
        plt.plot([mu[0], direction_2[0]], [mu[1], direction_2[1]], zs = contour_location, zdir = 'z', color = 'black')

    # Label the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Name the classes
    ax.text(7, 0, contour_location, 'Class-0', 'x')
    ax.text(-6, 7.6, contour_location, 'Class-2', 'y')
    ax.text(12, 5, contour_location, 'Class-1', 'x')

    # Save the figure
    plt.savefig(plot)

if __name__ == '__main__':
    plotPDF([np.array([1.0, 2.0]),\
             np.array([5.0, 7.0]),\
             np.array([-1.0, 11.0])],\
                
            [np.array([[1, 0.6], [0.6, 1]]),\
             np.array([[1.5, 0.3], [0.3, 1.5]]),\
             np.array([[1.2, 0], [0, 1.2]])],\

            'plots/pdf.jpg')
    plt.show()

