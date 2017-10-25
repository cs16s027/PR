import os
import numpy as np

def readMFCC(mfcc_file):
    seqs = [line.strip().split(' ') for line in open(mfcc_file, 'r').readlines()][1 : ]
    return np.float32(seqs)

def getCode(codes, vector):
    dists = []
    for code in codes:
        dists.append(np.sum(np.power(code - vector, 2)))
    return str(np.argmin(dists))

def codify(codes, seq, coded_seq_path):
    coded_seq_file = open(coded_seq_path, 'w')
    coded_seq_file.write(' '.join([getCode(codes, frame) for frame in seq]) + '\n')
    coded_seq_file.close()

if __name__ == '__main__':

    codes = np.load('data/proc/codes.npy')
    for stage in ['test1', 'test2']:
        for test in os.listdir('data/raw/%s' % stage):
            mfcc_file = os.path.join('data/raw/%s' % stage , test)
            seq = readMFCC(mfcc_file)
            coded_seq_path = 'data/proc/%s/%s.txt' % (stage, test.split('.mfcc')[0])
            codify(codes, seq, coded_seq_path)
            print 'Quantized %s in %s' % (test, stage)

