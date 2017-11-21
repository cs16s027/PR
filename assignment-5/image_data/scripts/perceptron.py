import h5py
import numpy as np
np.random.seed(0)

def y(x, w):
    y_ = np.dot(w, x)
    if y_ > 0:
        return 1
    return -1

def trainPerceptron(positive, negative):
    data = []
    for p in positive:
        p = np.append(p, 1)
        data.append((p, 1))
    for n in negative:
        n = np.append(n, 1)
        data.append((n, -1))
    np.random.shuffle(data)
    print len(data)
    w = np.zeros((data[0][0].shape[0]))


    epochs = 1
    for epoch in np.arange(1, epochs + 1, 1):
        error = 0.0
        for point in data:
            x, label = point
            pred = y(x, w) * label
            if pred <= 0:
                error += -pred
                w += label * x
        print 'Error after %s epochs = %s' % (epoch, error)
    return w
        

train = h5py.File('data/train.h5', 'r')
labels = ['0', '1', '2']
weights = dict()
for label in labels:
    plabel = label
    nlabels = list(set(labels) - set(label))
    positive = train[plabel][:]
    negative = np.concatenate([ train[nlabels[0]][:], train[nlabels[1][:]] ])
    np.random.shuffle(positive)
    np.random.shuffle(negative)
    positive = np.reshape(positive, (-1, 23))
    negative = np.reshape(negative, (-1, 23))
    print positive.shape, negative.shape
    weights[plabel] = trainPerceptron(positive, negative)
    
test = h5py.File('data/test.h5', 'r')
for label in labels:
    images = test[label][:]
    for image in images:
        pred_image = []
        for vector in image:
            pred_vector = []
            for plabel in labels:
                pred_vector.append(y(vector, weights[plabel]))
            pred_image.append(np.argmax(pred_vector))
            

