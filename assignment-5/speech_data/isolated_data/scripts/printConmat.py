import numpy as np
import sys

def printConfusionMatrix(predictions):
    mapping = {'1' : 0, '5' : 1, 'z' : 2}
    predictions = [line.strip().split() for line in open(predictions, 'r').readlines()]
    conmat = np.zeros((3, 3), dtype = np.int)
    for line in predictions:
        t_label = mapping[line[0]]
        p_label = np.argmax([float(word) for word in line[1 : ]])
        conmat[t_label, p_label] += 1
    print '\t0\t1\t2'
    for t_label in ['0', '1', '2']:
        print t_label + '\t' + '\t'.join([str(conmat[int(t_label), j]) for j in [0, 1, 2]])

printConfusionMatrix(sys.argv[1])
