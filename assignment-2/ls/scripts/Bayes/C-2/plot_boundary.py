from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from helpers import multivariateNormalPDF

import numpy as np

def plotBoundaries(mus, covs, plot, title, figure = 1):
    print 'Plotting boundary'
    # Initialize the matplotlib figure object
    fig = plt.figure(figure)
    # A 2d axes object
    ax = fig.gca()

    # Compute the gaussian for each class
    x, y = np.mgrid[-10 : 20 : 500j, -10 : 20 : 500j]
    xy = np.column_stack([x.flat, y.flat])
    zs = []
    for label in [0, 1, 2]:
        mu, cov = mus[label], covs[label]
        # Compute the gaussian on this grid with mean = mu, covariance = cov
        z = multivariateNormalPDF(xy, mu, cov)
        z = z.reshape(x.shape)
        # Populate Gaussians : zs
        zs.append(z)

    # Make a decision
    colors = np.array(['red', 'green', 'blue'])
    zs = np.argmax(np.array(zs), axis = 0)
    ax.scatter(x.flat, y.flat, color = colors[zs].flat)

    # Plot the line joining means
    ax.plot([mus[0][0], mus[1][0]], [mus[0][1], mus[1][1]], color = 'black')
    ax.plot([mus[1][0], mus[2][0]], [mus[1][1], mus[2][1]], color = 'black')
    ax.plot([mus[0][0], mus[2][0]], [mus[0][1], mus[2][1]], color = 'black')

     
    # Label the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    # Name the classes
    ax.text(6, 0, 'Class-0')
    ax.text(-6, 9.0, 'Class-2')
    ax.text(13, 6.5, 'Class-1')

    # Save the figure
    plt.title(title)
    plt.savefig(plot)

if __name__ == '__main__':
    plotBoundaries([np.array([1.0, 2.0]),\
             np.array([5.0, 7.0]),\
             np.array([-1.0, 11.0])],\
                
            [np.array([[1, 0.6], [0.6, 1]]),\
             np.array([[1.5, 0.3], [0.3, 1.5]]),\
             np.array([[1.2, 0], [0, 1.2]])],\
            'Test plot',\
            'plots/boundary.jpg')
    plt.show()

