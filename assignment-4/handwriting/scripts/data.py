import numpy as np
import h5py
from random import shuffle
import sys

def writeData(stage, stage_name, feat_indicator):
    labels = ['bA', 'dA', 'lA']
    for label in labels:
        np.save('data/proc/%s/%s_%s.npy' % (stage_name, label, feat_indicator), stage[label])

def extractFeatures(vector, feat_indicator):

    x = np.array([vector[i] for i in range(vector.shape[0]) if i % 2 == 0])
    y = np.array([vector[i] for i in range(vector.shape[0]) if i % 2 != 0])

    # Number of points
    N = x.shape[0]

    def getDerivatives(x, y):
        dx, dy = [], []
        for i in range(x.shape[0])[2 : -2]:
            dx_, dy_ = 0, 0
            for j in range(2):
                dx_ += j * (x[i + j] - x[i - j])
                dy_ += j * (y[i + j] - y[i - j])
            dx_ /= (2 * (1 * 1 + 2 * 2))
            dy_ /= (2 * (1 * 1 + 2 * 2))
            dx.append(dx_)
            dy.append(dy_)
        dx = np.array(dx)
        dy = np.array(dy)
        return dx, dy

    # First derivatives
    dx, dy = getDerivatives(x, y)
    
    # Second derivatives
    d2x, d2y = getDerivatives(dx, dy)

    # Curvature
    kt = []
    epsilon = 1e-9
    for i in range(d2x.shape[0]):
        kt.append( (dx[i + 2] * d2y[i] - d2x[i] * dy[i + 2]) / np.power(dx[i + 2] * dx[i + 2] + dy[i + 2] * dy[i + 2] + epsilon, 3.0 / 2.0) )
    kt = np.array(kt)

    # Assemble features
    feat_map = {'x' : x[ 4 : -4], 'dx' : dx[2 : -2], 'd2x' : d2x, 'y' : y[4 : -4], 'dy' : dy[2 : -2], 'd2y' : d2y, 'kt' : kt}
    features = []
    for i in range(kt.shape[0]):
        feats_ = []
        for f in feat_indicator.split(','):
            feats_.append(feat_map[f][i])
        features.append(feats_)

    return np.array(features)

if __name__ == '__main__':

    _, feat_indicator = sys.argv

    labels = ['bA', 'dA', 'lA']
    label_seqs = {}
    for label in labels:
        raw = [line.strip() for line in open('data/raw/%s.ldf' % label, 'r').readlines()]
        seqs = []
        for line in raw:
            if line == '1' or line == label:
                continue
            seq = np.array([float(word) for word in line.split()][1 : ])
            seqs.append(seq)
        seqs = np.array(seqs)
        label_seqs[label] = seqs

    label_feats = {}
    train_feats = {}
    test_feats = {}
    np.random.seed(0)
    for label in labels:
        seqs = label_seqs[label]
        label_feats[label] = []
        for seq in seqs:
            seq_features = extractFeatures(seq,feat_indicator)
            label_feats[label].append(seq_features)
        label_feats[label] = np.array(label_feats[label])
        np.random.shuffle(label_feats[label])

        size = len(label_feats[label])
        train_feats[label] = label_feats[label][ : int(size * 0.70)]
        test_feats[label] = label_feats[label][int(size * 0.70) : ]
        
    writeData(train_feats, 'train', feat_indicator)
    writeData(test_feats, 'test', feat_indicator)

    all_data = []
    for label in labels:
        print train_feats[label].shape
        print test_feats[label].shape
        for seq in train_feats[label]:
            all_data.append(seq)
    all_data = np.array(all_data)
    print all_data.shape
    np.save('data/proc/train/background_%s.npy' % feat_indicator, all_data)
