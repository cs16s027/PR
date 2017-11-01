import os
import h5py
import numpy as np
from random import shuffle


def readMFCC(mfcc_file):
    seqs = [line.strip().split(' ') for line in open(mfcc_file, 'r').readlines()][1 : ]
    return np.float32(seqs)

def writeData(stage, stage_name):
    for label in ['1', '5', 'z']:
        seqs = []
        for mfcc_file in stage[label]:
            seq = readMFCC(mfcc_file)
            seqs.append(seq)
        seqs = np.asarray(seqs)
        np.save('data/proc/%s/%s.npy' % (stage_name, label), seqs)
  
if __name__ == '__main__':

    np.random.seed(0)
    labels = os.listdir('data/raw')

    label_map, train_map, test_map = {}, {}, {}
    for label in labels:
        label_path = os.path.join('data/raw', label)
        mfcc_files = os.listdir(label_path)
        label_map[label], train_map[label], test_map[label] = [], [], []
        for mfcc_file in mfcc_files:
            mfcc_file_path = os.path.join(label_path, mfcc_file)
            label_map[label].append(mfcc_file_path)
        np.random.shuffle(label_map[label])
        print 'Number of files in %s = %s' % (label, len(label_map[label]))
        size = len(label_map[label])
        train_map[label] = label_map[label][ : int(size * 0.71)]
        test_map[label] = label_map[label][int(size * 0.71) : ]

    writeData(train_map, 'train')
    writeData(test_map, 'test')
    
