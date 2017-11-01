from dtw_solver import dtw
import sys

import numpy as np
from scipy import stats

_, feat_indicator = sys.argv
train = np.load('data/proc/train/background_%s.npy' % feat_indicator)
labels = ['bA', 'dA', 'lA']
label_map = {'bA' : '0', 'dA' : '1', 'lA' : '2'}
ground_truth = np.array(['bA'] * 67 + ['dA'] * 69 + ['lA'] * 68)
Ks = [1, 2, 3, 4, 5, 10, 15, 20, 40, 50, 80, 100, 120]

results = []
for K in Ks:
    results_ = open('results/dtw/dtw_%s.txt' % K, 'w')
    results_.write('%s\n' % K)
    results.append(results_)

for label in labels:
    print 'Testing class %s' % label
    test = np.load('data/proc/test/%s_%s.npy' % (label, feat_indicator))
    for index, test_seq in enumerate(test):
        print 'Testing sequence %s' % index
        dists = [dtw(train_seq, test_seq) for train_seq in train]
        for kindex, K in enumerate(Ks):
            top_k = ground_truth[np.argsort(dists)][ : K]
            prob_ = np.zeros((3, ))
            for index_, l in enumerate(labels):
                prob_[index_] = np.where(top_k == l)[0].shape[0]
            prob_ /= prob_.sum()
            prob_string = ' '.join([str(word) for word in prob_])
            results[kindex].write('%s %s\n' % (label_map[label], prob_string))

for i in range(len(Ks)):
    results[i].close()

