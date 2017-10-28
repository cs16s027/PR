from dtw_solver import dtw

import numpy as np
from scipy import stats

train = np.load('data/proc/train/dtw.npy')
labels = ['1', '5', 'z']
ground_truth = np.array(['1'] * 40 + ['5'] * 40 + ['z'] * 40)
predictions = []
K = 10

for label in labels:
    print 'Testing class %s' % label
    test = np.load('data/proc/test/%s.npy' % label)
    for index, test_seq in enumerate(test):
        print 'Testing sequence %s' % index
        dists = [dtw(train_seq, test_seq) for train_seq in train]
        top_k = ground_truth[np.argsort(dists)][ : K]
        predictions.append(stats.mode(top_k)[0][0])
        print 'Prediction for sequence %s is %s' % (index, predictions[-1])
        
correct = 0.0
for index, gt in enumerate(ground_truth):
    if gt == predictions[index]:
        correct += 1
print correct / 120.0
