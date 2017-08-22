# Script to do eigenvalue decomposition

import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

# EigenValueDecomposition
# This will decompose the input matrix A as: 
# A = Q _/\_ invQ
# Returns the three matrices Q, _/\_ and invQ
def doEVD(A):
    
    eig, Q = np.linalg.eig(A)

    try:
        invQ = np.linalg.inv(Q)
    except np.linalg.LinAlgError:
        print 'Exiting: The input matrix is not invertible'
        exit()

    return Q, eig, invQ

# A rank-n real, approximation of A
def approximate(Q, eig, invQ, n):	 
    dim = eig.shape[0]
    approxA = np.zeros((dim, dim), dtype = np.complex)

    for i in range(n):
        q_i = Q[:, i].reshape((dim, 1))
        invQ_i = invQ[i, :].reshape((1, dim))
        qinvQ_i = q_i * invQ_i
        approxA += eig[i] * qinvQ_i

    return np.real(approxA)

# Returns the Frobenius norm of A
def norm(A):
    row, col = A.shape
    val = np.float32(0.0)
    for x in range(row):
        for y in range(col):
            val += np.float32(A[x][y]) * np.float32(A[x][y])
    return np.sqrt(val)

def writeImages(approxA, errorA, path):
    row, col = approxA.shape
    pad = 10
    image = np.zeros((row, col * 2 + pad))
    image[ : , 0 : col] = approxA
    image[ : , col + pad : 2 * col + pad] = errorA 
    cv2.imwrite(path, image)

def plotEigval(eig, eigenplot):
    eig_index = range(len(eig) + 1)[1 : ]
    abs_eig = []
    for i in range(len(eig)):
        abs_eig.append(np.absolute(eig[i]))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale('log')
    ax.plot(eig_index, abs_eig)
    plt.xlim([0, 260])

    plt.xlabel('Eigen-index')
    plt.ylabel('Eigenvalue (log)')
    plt.title('Eigenvalue (log) .vs. Eigen-index')
    plt.savefig(eigenplot)

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print 'usage : python scripts/eigen.py <input-image> <rank> <output-image> <eigenplot>'
        print '<input-image> : path to input image'
        print '<rank> : order of approximation'
        print '<output-image> : path to output image'
        exit()
    _, input_image, rank, output_image, eigenplot = sys.argv

    # Read image as matrix A; normalize it
    A = cv2.imread(input_image, 0) / 255.0
    print 'The image is of shape', A.shape
    # Do Eigenvalue Decomposition of A
    Q, eig, invQ = doEVD(A)
    # Plot the eigenvalues of A
    plotEigval(eig, eigenplot)
    # Order of approximation
    rank = np.int(rank)
    # Construct the approximate matrix with given rank
    approxA = approximate(Q, eig, invQ, rank)
    # Compute the error matrix
    errorA = A - approxA
    # Compute the relative error in the approximation
    error = norm(errorA) / norm(A)
    print 'Relative error = %f' % error
    # Write the reconstructed images to disk; re-scale images before writing
    writeImages(approxA * 255.0, errorA * 255.0, output_image)

