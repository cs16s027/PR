import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

_, feat_indicator = sys.argv

window = 50
labels = ['bA', 'dA', 'lA']
C = 1.0

def softmax(vec):
    m = np.max(vec)
    s = np.sum(np.exp(vec - m))
    vec_ = []
    for v in vec:
        vec_.append(np.exp(v - m) / s)
    return vec_


TRAIN = True
TEST = True

if TRAIN == True:
    train_points = []
    for label in labels:
        data = np.load('data/proc/train/%s_%s.npy' % (label, feat_indicator))
        for seq in data:
            seql = seq.shape[0]
            if seql <= window + 10:
                r = np.abs(window - seql) + 5
                seq = np.concatenate([seq, np.zeros((r, 3))])
                seql = seq.shape[0]
            for start in np.arange(1, seql - window, 1):
                point = np.concatenate(seq[start : start + window])
                #point = np.mean(seq[start : start + window], axis = 0)
                train_points.append((point, label))

    np.random.shuffle(train_points)
    train_X, train_Y = zip(*train_points)
    clf = SVC(kernel = 'linear', C = C, verbose = True, max_iter = 100000)
    print train_X[0].shape
    print len(train_Y)
    clf.fit(train_X, train_Y)
    joblib.dump(clf, 'models/model_%s_%s.pkl' % (C, window))

if TEST == True:
    testf = open('results/%s_%s.txt' % (C, window), 'w')
    mapping = {0 : 'bA', 1 : 'dA', 2 : 'lA'}
    clf = joblib.load('models/model_%s_%s.pkl' % (C, window))
    count, correct = 0.0, 0.0
    for label in labels:
        data = np.load('data/proc/test/%s_%s.npy' % (label, feat_indicator))
        for seq in data:
            count += 1
            test_points = []
            seql = seq.shape[0]
            if seql <= window + 10:
                r = np.abs(window - seql) +5
                seq = np.concatenate([seq, np.zeros((r, 3))])
                seql = seq.shape[0]
            for start in np.arange(1, seql - window, 1):
                point = np.concatenate(seq[start : start + window])
                #point = np.mean(seq[start : start + window], axis = 0)
                test_points.append((point, label))
            test_X, _ = zip(*test_points)
            df = clf.decision_function(test_X)
            pred =  np.mean(df, axis = 0)
            probs = softmax(pred)
            pred_label = mapping[np.argmax(pred)]
            print label, mapping[np.argmax(pred)]
            testf.write(' '.join([label] + [str(prob) for prob in probs]) + '\n')
    
            if label == pred_label:
                correct += 1
    testf.close()
    print correct / count * 100
