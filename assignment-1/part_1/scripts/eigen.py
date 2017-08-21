# Script to do eigenvalue decomposition

import sys
import cv2
import numpy as np

# EigenValueDecomposition
# This will decompose the input matrix A as: 
# A = Q _/\_ invQ
# Returns the three matrices Q, _/\_ and invQ
def doEVD(A):
    
    eig, Q = np.linalg.eig(input_image)

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

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print 'usage : python scripts/eigen.py <input-image> <rank>'
        exit()
    _, input_image, rank = sys.argv
    input_image = cv2.imread(input_image, 0) / 255.0
    rank = np.int(rank)
    print 'The image is of shape', input_image.shape

    Q, eig, invQ = doEVD(input_image)

    approxA = approximate(Q, eig, invQ, rank) * 255.0

    cv2.imwrite('recon.jpg', approxA)
    
