import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

means = 5
labels = ['1', '5', 'z']
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
        data = np.load('data/proc/train/%s.npy' % label)
        for seq in data:
            train_points.append((seq, label))

    np.random.shuffle(train_points)
    train_X, train_Y = zip(*train_points)
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    print train_X.shape, train_Y.shape
    clf = SVC(kernel = 'linear', C = C, verbose = True)
    clf.fit(train_X, train_Y)
    joblib.dump(clf, 'models/model_%s_%s.pkl' % (C, means))

if TEST == True:
    testf = open('results/%s_%s.txt' % (C, means), 'w')
    mapping = {0 : '1', 1 : '5', 2 : 'z'}
    clf = joblib.load('models/model_%s_%s.pkl' % (C, means))
    count, correct = 0.0, 0.0
    for label in labels:
        data = np.load('data/proc/test/%s.npy' % label)
        for seq in data:
            count += 1
            test_X = np.reshape(seq, (1, -1))
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
