import mayavi.mlab as mlab

import numpy as np
from scipy.stats import multivariate_normal

def plotPDF(mus, covs, plot):
    gmin, gmax = np.min(np.array(mus)), np.max(np.array(mus))
    # Generate the underlying x-y grid
    x, y = np.mgrid[gmin - 7 : gmax + 7 : 100j, gmin - 7 : gmax + 7 : 100j]
    # Generate tuples (x, y) : X x Y
    xy = np.column_stack([x.flat, y.flat])
    # Base plot 
    base = np.zeros_like(x)
    #mlab.surf(x, y, base)

    # Compute the gaussian for each class
    zs = []
    for label in [0, 1, 2]:
        mu, cov = mus[label], covs[label]
        # Compute the gaussian on this grid with mean = mu, covariance = cov
        z = multivariate_normal.pdf(xy, mean = mu, cov = cov)
        # Reshape the points to fall in line with x,y 
        z = z.reshape(x.shape)
        # Populate Gaussians : zs
        zs.append(z)
    
    zmin, zmax = np.min(np.array(zs)), np.max(np.array(zs))
    contour_location = zmin - 2 * (zmax - zmin)
    extent = [gmin, gmax, gmin, gmax, zmin, zmax]
    # Plot the pdf and the contours
    for label in [0, 1, 2]:
        # Surface plot
        mlab.surf(x, y, zs[label], warp_scale = 'auto')
        # Contour plot 
        mlab.contour_surf(zs[label])

    mlab.show()

    # Plot the directions
    '''
    _, eigvec = np.linalg.eig(cov)
    eig_extent = np.max(x)
    direction_1 = mu + eig_extent * eigvec[:, 0]
    direction_2 = mu + eig_extent * eigvec[:, 1]
    plt.plot([mu[0], direction_1[0]], [mu[1], direction_1[1]], zs = contour_location, zdir = 'z', color = 'black')
    plt.plot([mu[0], direction_2[0]], [mu[1], direction_2[1]], zs = contour_location, zdir = 'z', color = 'black')
    '''

if __name__ == '__main__':
    plotPDF([np.array([1.0, 2.0]),\
             np.array([5.0, 7.0]),\
             np.array([-1.0, 11.0])],\
                
            [np.array([[1, 0.6], [0.6, 1]]),\
             np.array([[1.5, 0.3], [0.3, 1.5]]),\
             np.array([[1.2, 0], [0, 1.2]])],\

            'plots/pdf.jpg')
    mlab.show()
