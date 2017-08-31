# Script to do eigenvalue decomposition

import sys
import cv2
import numpy as np
np.random.seed(1)
from matplotlib import pyplot as plt

# EigenValueDecomposition
# This will decompose the input matrix A as: 
# A = Q _/\_ invQ
# Returns the three matrices Q, _/\_ and invQ
def doEVD(A):
    
    eig, Q = np.linalg.eig(A)
    ids = eig.argsort()[::-1]
    eig = eig[ids]
    Q = Q[:, ids]

    try:
        invQ = np.linalg.inv(Q)
    except np.linalg.LinAlgError:
        print 'Exiting: The input matrix is not invertible'
        exit()

    return Q, eig, invQ

# Check if matrices A and B are orthogonal
def check(A, B):
    X = np.dot(A, B)
    try:
        assert np.array_equal(np.rint(X), np.eye(A.shape[0]))
        return True
    except AssertionError:
        print 'The matrices are not orthogonal'
        return False

# Do an SVD on A
def doSVD(A):
    print 'Shape of matrix A =', A.shape
    # Get the symmetric matrix A'A
    Asym = np.dot(A.T, A)
    print 'Shape of A\'A =', Asym.shape
    # Get the EVD of A'A
    V, eig, Vt = doEVD(Asym)
    # Check if V and Vt are orthogonal
    check(V, Vt)
    # Number of eigenvalues
    print 'Number of eigenvalues of A\'A =', eig.shape[0]
    # Get singular values
    sings = np.sqrt(eig)
    # Use A * V = U * sings, to get U
    U = np.dot(A, V) / sings
    # Check if U and U' are orthogonal
    check(U.T, U)
    # Singular matrix
    sing_matrix = np.diag(sings)
    # A = U * sing_matrix * Vt
    return U, sings, Vt

# A rank-n real, approximation of A
def approximate(U, sings, Q, n, random = False):
    dimX, dimY = U.shape[0], Q.shape[1]
    dim = sings.shape[0]
    approxA = np.zeros((dimX, dimY))

    indices = range(dim)
    
    if random == True:
        np.random.shuffle(indices)

    for index in range(n):
        i = indices[index]
        u_i = U[:, i].reshape((dimX, 1))
        q_i = Q[i, :].reshape((1, dimY))
        uq_i = u_i * q_i
        approxA += sings[i] * uq_i

    return approxA

# Returns the Frobenius norm of A
def norm(A):
    row, col = A.shape
    val = np.float32(0.0)
    for x in range(row):
        for y in range(col):
            val += np.float32(A[x][y]) * np.float32(A[x][y])
    return np.sqrt(val)

def writeImages(A, approxA, errorA, rank, error, random, path):
    row, col = approxA.shape
    pad = 10
    image = np.zeros((row + 500, 3 * col + 2 * pad))
    image[0 : row, 0 : col] = A
    image[0 : row , col + pad : 2 * col + pad] = approxA
    image[0 : row, 2 * col + 2 * pad : 3 * col + 2 * pad] = errorA 
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Orig', (800, col + 40), font, 1, (255,255,255), 2)
    cv2.putText(image, 'Recon', (1600, col + 40), font, 1, (255,255,255), 2)
    cv2.putText(image, 'Error', (2400, col + 40), font, 1, (255,255,255), 2)
    cv2.putText(image, '# Singular-values = %d' % rank, (220, col + 80), font, 1, (255, 255, 255), 2)
    cv2.putText(image, 'Rel Error = %s' % str(error)[:4], (250, col + 130), font, 1, (255, 255, 255), 2)
    cv2.putText(image, 'Random = %s' % random, (250, col + 170), font, 1, (255, 255, 255), 2)   

    cv2.imwrite(path, image)


def plotSingularval(sings, singularplot):
    sings_index = range(len(sings) + 1)[1 : ]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale('log')
    ax.plot(sings_index, sings)
    plt.xlim([0, sings.shape[0]])

    plt.xlabel('Singular-index')
    plt.ylabel('Singular-value (log)')
    plt.title('Singular-value (log) .vs. Singular-index')
    plt.savefig(singularplot)

def plotErrorDecay(ranks, errors, errorplot):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ranks, errors)
    plt.xlim([0, ranks[-1]])

    plt.xlabel('Singular-index')
    plt.ylabel('Relative-Error')
    plt.title('Relative-Error .vs. Singular-index')
    plt.savefig(errorplot)

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print 'usage : python scripts/basiEVD.py <input-image> <singularplot> <errorplot> <random>'
        print '<input-image> : path to input image'
        print '<singularplot> : path to save plot of singularvalues'
        print '<errorplot> : path to save plot of errors'
        print '<random> : Y or N, if Y it will take random singularvalues'
        exit()
    _, input_image, singularplot, errorplot, random = sys.argv
    random = True if random == 'Y' else False

    # Read image as matrix A; normalize it
    A = cv2.imread(input_image, 0) / 255.0
    print 'The image is of shape', A.shape
    # Work with a smaller A'A matrix
    transpose_flag = False
    if A.shape[0] < A.shape[1]:
        A = A.T
        transpose_flag = True
    print 'The matrix A is of shape', A.shape
    # Do Singular Value Decomposition of A
    U, sings, Q = doSVD(A)
    # Plot the singularvalues of A
    plotSingularval(sings, singularplot)
    # Ranks to reconstruct
    ranks = np.arange(5, sings.shape[0], int(0.01 * sings.shape[0]))
    print len(ranks)
    errors = []
    for index, rank in enumerate(ranks):
        print index
        # Construct the approximate matrix with given rank
        approxA = approximate(U, sings, Q, rank, random)
        # Compute the error matrix
        errorA = A - approxA
        # Compute the relative error in the approximation
        error = norm(errorA) / norm(A)
        # Populate errors list
        errors.append(error)
    plotErrorDecay(ranks, errors, errorplot) 
    
