import h5py
import numpy as np
np.random.seed(0)

def y(x, w):
    y_ = np.dot(w, x)
    if y_ > 0:
        return 1
    return -1

def validate(test, weights):
    correct = 0.0
    count = 0.0
    for label in labels:
        for image in test[label][:]:
            count += 1
            image_vec = []
            for vec in image:
                vec_ = np.append(vec, 1)
                vec_vec = []
                for plabel in labels: 
                    vec_vec.append(np.dot(weights[plabel], vec_))
                image_vec.append(vec_vec)
            image_vec = np.array(image_vec)
            image_pred = np.mean(image_vec, axis = 0)
            pred_label = np.argmax(image_pred)
            if int(label) == int(pred_label):
                correct += 1
    return correct / count * 100

def trainPerceptron(train, valid):
    labels = ['0', '1', '2']
    weights = {'0' : '', '1' : '', '2' : ''}
    D = train[0][0].shape[0]
    for label in labels:
        weights[label] = np.zeros(D)
    epochs = 70
    eta = 0.001
    max_score = 0.0
    for epoch in np.arange(1, epochs + 1, 1):
        error = 0.0
        for point in train:
            ys = []
            x, label = point
            for plabel in labels:
                ys.append(np.dot(weights[plabel], x))
            pred = np.argmax(ys)
            if pred != int(label):
                pred_ = str(pred)
                weights[pred_] -= eta * x
                weights[label] += eta * x
                error += ys[pred]
        score = validate(valid, weights)
        if score > max_score:
            max_score = score
            print max_score
            wfile = h5py.File('models/best.h5', 'w')
            for label in labels: 
                wfile.create_dataset(data = weights[label], name = label, shape = weights[label].shape, dtype = weights[label].dtype)
            wfile.close()
        print 'Error after %s epochs = %s' % (epoch, error)
    return weights

train = h5py.File('data/train.h5', 'r')
valid = h5py.File('data/valid.h5', 'r')
labels = ['0', '1', '2']
data = []
for label in labels:
    train_ = np.reshape(train[label][:], (-1, 23))
    for vec in train_:
        vec_ = np.append(vec, 1)
        data.append((vec_, label))
np.random.shuffle(data)
weights = trainPerceptron(data, valid)

test = h5py.File('data/test.h5', 'r')
weights_ = h5py.File('models/best.h5', 'r')
weights = {'0' : weights_['0'][:], '1' : weights_['1'][:], '2' : weights_['2'][:]}
print validate(test, weights)
