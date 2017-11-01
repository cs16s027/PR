import os
import numpy as np
import sys

_, symbols, states = sys.argv
for data in ['1', '5', 'z']:
    for model in ['1', '5', 'z', 'background']:
        os.system('./scripts/hmm/test_hmm data/proc/test/%s_%s.seq models/hmm/%s_%s_%s.hmm | grep "alpha for" | cut -d"=" -f2 | cut -d" " -f2 > results/hmm/%s_%s.pred' % (data, symbols, model, symbols, states, model, data))
    os.system('paste -d" " results/hmm/1_%s.pred results/hmm/5_%s.pred results/hmm/z_%s.pred results/hmm/background_%s.pred > results/hmm/%s.pred' % (data, data, data, data, data))


label_map = {'1' : '0', '5' : '1' , 'z' : '2'}
results = open('results/hmm/%s_%s.txt' % (symbols, states), 'w')
for label in ['1', '5', 'z']:
    preds = [line.strip().split() for line in open('results/hmm/%s.pred' % label, 'r').readlines()]
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
