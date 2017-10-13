import numpy as np
from helpers import loadData, cluster

# For class 'i'

label = 0
data = loadData('data/normal/train.h5')[label]
K = 12

print 'Data :', data.shape

N, D = data.shape
epsilon = 0.0
# EM - 0th step
mus, assignments = cluster(data, K)
sigmas = np.zeros((K, D, D))
for k in range(K):
    points_in_k = data[np.where(assignments == k)]
    sigmas[k, :, :] = np.dot((points_in_k - mus[k]).T, points_in_k - mus[k]) / points_in_k.shape[0]
pis = np.array([np.sum(assignments == k) for k in range(K)], np.float32)
assert np.sum(pis) == N
pis /= N

print 'Means :', mus.shape
print 'Covariances :', sigmas.shape
print 'Mixture priors :', pis.shape

for i in range(5):
    print 'Iteration %s of EM' % str(i + 1)
    # E Step
    print '######### E-step #########'
    print 'Necessary computations for calculating the likelihood'
    invs, dets = zip(*[[np.linalg.inv(sigma), np.linalg.det(sigma)] for sigma in sigmas])
    factors = [1.0 / np.sqrt(np.power(2 * np.pi, D) * dets[k]) for k in range(K)]
    invs, dets, factors = np.array(invs), np.array(dets), np.array(factors)

    print 'Computing the responsibilities'
    gamma = np.zeros((N, K))
    for n in range(data.shape[0]):
        x = data[n]
        exponents = -0.5 * np.array([np.dot(np.dot((x - mus[k]).T, invs[k]), (x - mus[k]))\
                                    for k in range(K)])
        exp_diffs = np.zeros((K, K))
        for k_1 in range(K):
            for k_2 in range(K)[k_1 + 1 : ]:
                exp_diffs[k_1, k_2] = exponents[k_1] - exponents[k_2]
        exp_diffs[np.where(exp_diffs < -2)] = 0.0
        exp_diffs[np.where(exp_diffs > 2)] = np.inf
        exp_diffs -= exp_diffs.T
        for k in range(K):
            gamma[n][k] = 1 / np.sum( np.power(np.e, exp_diffs[:, k]) * (factors / factors[k]) * (pis /pis[k]) )
    
    print '####### M-step ########'
    # M Step
    N_eff = np.sum(gamma, axis = 0)
    mus = (np.dot(gamma.T, data).T / N_eff).T
    sigmas = np.zeros((K, D, D))
    for k in range(K):
        for n in range(data.shape[0]):
            x = data[n]
            sigmas[k, :, :] += gamma[n][k] * np.dot((x - mus[k]).reshape((D, 1)), (x - mus[k]).reshape((1, D)))
        sigmas[k, :, :] /= N_eff[k]
    #assert np.rint(np.sum(N_eff)) == N
    print N, np.sum(N_eff)
    pis = N_eff / N

