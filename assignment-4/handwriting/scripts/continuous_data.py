import os
from scipy.io import loadmat
import matplotlib.pyplot as plt

import numpy as np
import h5py
from random import shuffle
import sys

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

def extractFeatures(x, y, feat_indicator):

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

def dataPlot():
    for i in range(x.shape[0]):
        plt.scatter(x[i], y[i])

    plt.savefig('plots/viz/%s.jpg' % f)
    plt.clf()



for f in ['1', '2', '3']:

    data_ = loadmat('data/cont/%s.mat' % f)
    x = data_['xData'][0]
    y = data_['yData'][0]

    feats = extractFeatures(x, y, 'kt')
    np.save('data/cont/%s.npy' % f, feats)


    feat_indicator, symbols = 'kt', 16
    codes = np.load('data/proc/codes_%s_%s.npy' % (feat_indicator, symbols))
    codify(codes, [feats], 'data/cont/%s.seq' % f)

datas = ['data/cont/1.seq', 'data/cont/2.seq', 'data/cont/3.seq']
models = os.listdir('models/cont')

for data_file in datas:
    predictions = []
    for model in models:
        model_file = 'models/cont/%s' % model
        os.system('./scripts/hmm/test_hmm %s %s | grep "alpha for" | cut -d"=" -f2 | cut -d" " -f2 > temp' % (data_file, model_file))
        predictions.append(float(open('temp', 'r').readlines()[0].strip()))
        os.system('rm temp')
    print data_file.split('/')[2].split('.seq')[0], models[np.argmax(predictions)].split('.hmm')[0]

