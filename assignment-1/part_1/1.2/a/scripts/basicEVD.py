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
    ids = np.absolute(eig).argsort()[::-1]
    eig = eig[ids]
    Q = Q[:, ids]

    try:
        invQ = np.linalg.inv(Q)
    except np.linalg.LinAlgError:
        print 'Exiting: The input matrix is not invertible'
        exit()

    '''
    real_eig = []
    real_Q = []
    real_invQ = []
    for i in range(eig.shape[0]):
        if np.isreal(eig[i]):
            real_eig.append(np.real(eig[i]))
            real_Q.append(np.real(Q[:, i]))
            real_invQ.append(np.real(invQ[i, :]))
    
    real_eig, real_Q, real_invQ = np.array(real_eig), np.array(real_Q).T, np.array(real_invQ)
    print real_eig.shape, real_Q.shape, real_invQ.shape
    return real_Q, real_eig, real_invQ
    '''
    return Q, eig, invQ

# A rank-n real, approximation of A
def approximate(Q, eig, invQ, n, random = False):
    dim = Q.shape[0]
    approxA = np.zeros((dim, dim), dtype = np.complex)

    eig_count = eig.shape[0]
    indices = range(eig_count)
    #np.random.seed(1)
    if random == True:
        np.random.shuffle(indices)

    for index in range(n):
        i = indices[index]
        q_i = Q[:, i].reshape((dim, 1))
        invQ_i = invQ[i, :].reshape((1, dim))
        qinvQ_i = np.dot(q_i, invQ_i)
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
    cv2.putText(image, '# Eigenvalues = %d' % rank, (220, col + 80), font, 1, (255, 255, 255), 2)
    cv2.putText(image, 'Rel Error = %s' % str(error)[:4], (250, col + 130), font, 1, (255, 255, 255), 2)
    cv2.putText(image, 'Random = %s' % random, (250, col + 170), font, 1, (255, 255, 255), 2)   

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
    plt.xlim([0, len(eig)])

    plt.xlabel('Eigen-index')
    plt.ylabel('Eigenvalue (log)')
    plt.title('Eigenvalue (log) .vs. Eigen-index')
    plt.savefig(eigenplot)

def plotErrorDecay(ranks, errors, errorplot):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ranks, errors)
    plt.xlim([0, ranks[-1]])

    plt.xlabel('Eigen-index')
    plt.ylabel('Relative-Error')
    plt.title('Relative-Error .vs. Eigen-index')
    plt.savefig(errorplot)

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print 'usage : python scripts/basiEVD.py <input-image> <rank> <output-image> <eigenplot> <random>'
        print '<input-image> : path to input image'
        print '<rank> : order of approximation'
        print '<output-image> : path to output image'
        print '<eigenplot> : path to save plot of eigenvalues'
        print '<errorplot> : path to save plot of errors'
        print '<random> : Y or N, if Y it will take random eigenvalues'
        exit()
    _, input_image, rank, output_image, eigenplot, errorplot, random = sys.argv

    # Read image as matrix A; normalize it
    A = cv2.imread(input_image, 0) / 255.0
    print 'The image is of shape', A.shape
    # Do Eigenvalue Decomposition of A
    Q, eig, invQ = doEVD(A)
    # Plot the eigenvalues of A
    plotEigval(eig, eigenplot)
    # Order of approximation
    rank = np.int(rank)
    # Random N eigenvalues
    random = True if random == 'Y' else False
    # Construct the approximate matrix with given rank
    approxA = approximate(Q, eig, invQ, rank, random)
    # Compute the error matrix
    errorA = A - approxA
    # Compute the relative error in the approximation
    error = norm(errorA) / norm(A)
    print 'Relative error = %f' % error
    # Write the reconstructed images to disk; re-scale images before writing
    writeImages(A * 255.0, approxA * 255.0, errorA * 255.0, rank, error, random, output_image)

    ##############################################################################################33
    ranks = np.arange(1, 256, 1)
    errors = []
    for rank in ranks:
        # Construct the approximate matrix with given rank
        approxA = approximate(Q, eig, invQ, rank, random)
        # Compute the error matrix
        errorA = A - approxA
        # Compute the relative error in the approximation
        error = norm(errorA) / norm(A)
        # Populate errors list
        errors.append(error)

    plotErrorDecay(ranks, errors, errorplot) 
    
