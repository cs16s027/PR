import numpy as np
from sklearn.svm import SVC
window = 5
labels = ['1', '5', 'z']

train_points = []
for label in labels:
    data = np.load('data/proc/train/%s.npy' % label)
    for seq in data:
        seql = seq.shape[0]
        for start in np.arange(1, seql - window, 1):
            point = np.concatenate(seq[start : start + window])
            train_points.append((point, label))

np.random.shuffle(train_points)
train_X, train_Y = zip(*train_points)
clf = SVC()
clf.fit(train_X, train_Y)

test_points = []
for label in labels:
    data = np.load('data/proc/test/%s.npy' % label)
    for seq in data:
        seql = seq.shape[0]
        for start in np.arange(1, seql - window, 1):
            point = np.concatenate(seq[start : start + window])
            test_points.append((point, label))

test_X, test_Y = zip(*test_points)
for i in range(len(test_X)):
    print test_Y[i], clf.predict(test_X[i])

