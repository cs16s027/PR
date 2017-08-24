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

# Check if matrices A and B are orthogonal
def check(A, B):
    X = np.matmul(A, B)
    try:
        assert np.array_equal(np.rint(X), np.eye(A.shape[0]))
        return True
    except AssertionError:
        print 'The matrices are not orthogonal'
        return False

# Do an SVD on A
def doSVD(A):
    thresh = 1e-6
    V, eig, Vt = doEVD(np.dot(A.T, A))
    V, eig, Vt = np.real(V), np.absolute(np.real(eig)), np.real(Vt)
    check(V, Vt)
    lamb = np.sqrt(eig)
    U = []
    for i in range(V.shape[0]):
        if lamb[i] > thresh:
            U.append(np.matmul(A, V) / lamb[i])
    U = np.array(U).T
    #U = np.matmul(A, V) / lamb
    check(U, U.T)
    exit()
    return U, lamb, Vt

# A rank-n real, approximation of A
def approximate(U, lambdas, Q, n, random = False):
    dimX, dimY = U.shape[0], Q.shape[1]
    dim = lambdas.shape[0]
    approxA = np.zeros((dimX, dimY))

    indices = range(dim)
    #np.random.seed(1)
    if random == True:
        np.random.shuffle(indices)

    for index in range(n):
        i = indices[index]
        u_i = U[:, i].reshape((dimX, 1))
        q_i = Q[i, :].reshape((1, dimY))
        uq_i = q_i * q_i
        approxA += lambdas[i] * uq_i

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
    image = np.zeros((row + 200, 3 * col + 2 * pad))
    image[0 : row, 0 : col] = A
    image[0 : row , col + pad : 2 * col + pad] = approxA
    image[0 : row, 2 * col + 2 * pad : 3 * col + 2 * pad] = errorA 
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Orig', (100, col + 40), font, 1, (255,255,255), 2)
    cv2.putText(image, 'Recon', (340, col + 40), font, 1, (255,255,255), 2)
    cv2.putText(image, 'Error', (620, col + 40), font, 1, (255,255,255), 2)
    cv2.putText(image, '# Singular-values = %d' % rank, (220, col + 80), font, 1, (255, 255, 255), 2)
    cv2.putText(image, 'Rel Error = %s' % str(error)[:4], (250, col + 130), font, 1, (255, 255, 255), 2)
    cv2.putText(image, 'Random = %s' % random, (250, col + 170), font, 1, (255, 255, 255), 2)   

    cv2.imwrite(path, image)


def plotSingularval(eig, singularplot):
    eig_index = range(len(eig) + 1)[1 : ]
    abs_eig = []
    for i in range(len(eig)):
        abs_eig.append(np.absolute(eig[i]))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale('log')
    ax.plot(eig_index, abs_eig)
    plt.xlim([0, 260])

    plt.xlabel('Singular-index')
    plt.ylabel('Singularvalue (log)')
    plt.title('Singularvalue (log) .vs. Singular-index')
    plt.savefig(singularplot)

def plotErrorDecay(ranks, errors, errorplot):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ranks, errors)
    plt.xlim([0, 260])

    plt.xlabel('Singular-index')
    plt.ylabel('Relative-Error')
    plt.title('Relative-Error .vs. Singular-index')
    plt.savefig(errorplot)

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print 'usage : python scripts/basiEVD.py <input-image> <rank> <output-image> <singularplot> <random>'
        print '<input-image> : path to input image'
        print '<rank> : order of approximation'
        print '<output-image> : path to output image'
        print '<singularplot> : path to save plot of singularvalues'
        print '<errorplot> : path to save plot of errors'
        print '<random> : Y or N, if Y it will take random singularvalues'
        exit()
    _, input_image, rank, output_image, singularplot, errorplot, random = sys.argv

    # Read image as matrix A; normalize it
    A = cv2.imread(input_image, 0) / 255.0
    print 'The image is of shape', A.shape
    # Do Singular Value Decomposition of A
    U, lamb, Q = doSVD(A)
    # Plot the singularvalues of A
    plotSingularval(lamb, singularplot)
    # Order of approximation
    rank = np.int(rank)
    # Random N singularvalues
    random = True if random == 'Y' else False
    # Construct the approximate matrix with given rank
    approxA = approximate(U, lamb, Q, rank, random)
    # Compute the error matrix
    errorA = A - approxA
    # Compute the relative error in the approximation
    error = norm(errorA) / norm(A)
    print 'Relative error = %f' % error
    # Write the reconstructed images to disk; re-scale images before writing
    writeImages(A * 255.0, approxA * 255.0, errorA * 255.0, rank, error, random, output_image)

    ranks = np.arange(5, 250, 5)
    errors = []
    for rank in ranks:
        # Construct the approximate matrix with given rank
        approxA = approximate(U, lamb, Q, rank, random)
        # Compute the error matrix
        errorA = A - approxA
        # Compute the relative error in the approximation
        error = norm(errorA) / norm(A)
        # Populate errors list
        errors.append(error)

    plotErrorDecay(ranks, errors, errorplot) 
    
