import os
import numpy as np
import sys

_, feat_indicator, symbols, states = sys.argv
for data in ['bA', 'dA', 'lA']:
    for model in ['bA', 'dA', 'lA', 'background']:
        os.system('./scripts/hmm/test_hmm data/proc/test/%s_%s_%s.seq models/hmm/%s_%s_%s_%s.hmm | grep "alpha for" | cut -d"=" -f2 | cut -d" " -f2 > results/hmm/%s_%s_%s.pred' % (data, feat_indicator, symbols, model, feat_indicator, symbols, states, model, feat_indicator, data))
    os.system('paste -d" " results/hmm/bA_%s_%s.pred results/hmm/dA_%s_%s.pred results/hmm/lA_%s_%s.pred results/hmm/background_%s_%s.pred > results/hmm/%s_%s.pred' % (feat_indicator, data, feat_indicator, data, feat_indicator, data, feat_indicator, data, feat_indicator, data))


label_map = {'bA' : '0', 'dA' : '1' , 'lA' : '2'}
results = open('results/hmm/%s_%s_%s.txt' % (feat_indicator, symbols, states), 'w')
for label in ['bA', 'dA', 'lA']:
    preds = [line.strip().split() for line in open('results/hmm/%s_%s.pred' % (feat_indicator, label), 'r').readlines()]
    for point in preds:
        lhoods = np.float32(point)
        #for i in range(3):
        #    lhoods[i] -= lhoods[3]
        lhoods = lhoods[ : -1]
        probs = np.zeros((3, ))
        l_matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                l_matrix[i, j] = lhoods[i] - lhoods[j] 
        for i in [0, 1, 2]:
            probs[i] = 1 / np.sum(np.exp(l_matrix[ :, i]))
        string = label_map[label] + ' ' + ' '.join([str(word) for word in probs])
        results.write(string + '\n')
results.close()

os.system('rm results/hmm/*.pred alphaout')
