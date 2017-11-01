import numpy as np
import sys
import os

def getCode(codes, vector):
    dists = []
    for code in codes:
        dists.append(np.sum(np.power(code - vector, 2)))
    return str(np.argmin(dists))

def codify(codes, seqs, coded_seqs_path):
    coded_seqs_file = open(coded_seqs_path, 'w')
    for seq in seqs:
        coded_seqs_file.write(' '.join([getCode(codes, frame) for frame in seq]) + '\n')
    coded_seqs_file.close()

if __name__ == '__main__':

    _, feat_indicator, symbols = sys.argv
    codes = np.load('data/proc/codes_%s_%s.npy' % (feat_indicator, symbols))

    for stage_name in ['train', 'test']:
        for label in ['bA', 'dA', 'lA', 'background']:
            if stage_name == 'test' and label == 'background':
                continue
            seqs_path = 'data/proc/%s/%s_%s.npy' % (stage_name, label, feat_indicator)
            coded_seqs_path = 'data/proc/%s/%s_%s_%s.seq' % (stage_name, label, feat_indicator, symbols)
            if os.path.isfile(coded_seqs_path):
                continue
            seqs = np.load(seqs_path)
            codify(codes, seqs, coded_seqs_path)
            print 'codified %s in %s' % (label, stage_name)

