import itertools
import numpy as np
from matplotlib import pyplot as plt

def loadData(datafile):
    lines = [line.strip().split(' ') for line in open(datafile, 'r').readlines()]
    dimX = [int(lines[0][1]), int(lines[0][0]) - 1]
    dimY = int(lines[0][1])

    lines.pop(0)
    
    X = np.zeros(dimX)
    Y = np.zeros(dimY)
    for index, line in enumerate(lines):
        Y[index] = np.float(line.pop())
        X[index, :] = np.array([np.float(word) for word in line])
        
    print 'X is of shape', X.shape
    print 'Y is of shape', Y.shape

    return X, Y

# Get all kth order terms for a polynomial in N variables
def getTerms(N, k):
    return [sorted(p) for p in itertools.product(range(N), repeat = k)]

# Get polynomial features of order n
def getFeatures(X, n):
    dim = X.shape[1]
    features = []
    for i, point_i in enumerate(X):
        features_i = []
        for order in np.arange(1, n + 1, 1):
            terms = getTerms(dim, order)
            for term in terms:
                features_i.append(np.product(point_i[term]))
        features.append(features_i)
    features = np.array(features)
    print 'Polynomial features are of shape', features.shape
    return features

def partitionData(data):
    X, Y = data
    dim = X.shape[0]
    
    # Shuffle the indices
    indices = np.arange(dim)
    np.random.seed(0)
    np.random.shuffle(indices)

    # Train, Validation and Test split
    split = [0.7, 0.2, 0.1]
    split_dict = {0 : 'Train', 1 : 'Validation', 2 : 'Test'}
        
    # Data will store the train, validation and test splits
    data = []
    start_index = 0
    for i in range(3):
        points_count = int(split[i] * dim)
        indices_i = indices[start_index : start_index + points_count]
        data_i = [X[indices_i], Y[indices_i]]
        print split_dict[i], 'data is of size', data_i[0].shape, data_i[1].shape
        data.append(data_i)

    return data

# Stochastic gradient descent
def train(train_data, valid_data):
    # Data
    train_X, train_Y = train_data
    train_size = train_X.shape[0]
    print '\n### Data ###'
    print 'Training on %d data-points' % train_size
    valid_X, valid_Y = valid_data
    valid_size = valid_X.shape[0]
    print 'Validating on %d data-points' % valid_size

    # Variable initialization
    params_dim = train_X.shape[1]
    print '\n### Variable Initialization ###'
    print 'Number of parameters = %d' % params_dim
    beta = np.zeros(params_dim)
    print 'Zero initialization'

    # Hyperparameters
    epochs = 100
    batch_size = 16
    lr = 0.001
    patience = 5
    print '\n### Hyperparameters ###'
    print 'Training for %d epochs with a batch size of %d and learning rate of %f' % (epochs, batch_size, lr)
    print 'Early stopping has been enabled with patience set to %d' % patience
    
    # Training
    print '\n### Training and Validation ###'
    validation_loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = int(float(train_size) / batch_size)
        for batch in range(batches):
            indices = np.arange(batch * batch_size, (batch + 1) * batch_size, 1)
            batch_loss = 0.0
            batch_loss_grad = np.zeros_like(beta)
            for i in indices:
                X_i, Y_i = train_X[i, :], train_Y[i]
                Y_hat_i = np.dot(X_i, beta)
                L_i = Y_i - Y_hat_i
                batch_loss += 0.5 * L_i * L_i
                batch_loss_grad += L_i * X_i 
            beta += lr * batch_loss_grad
        epoch_loss += batch_loss / train_size 
        print 'Training loss after epoch %d = %f' % ((epoch + 1), epoch_loss)
    
        # Validation
        if (epoch + 1) % 5 == 0:
            batch_size = 1
            epoch_loss = 0.0
            batches = int(float(valid_size) / batch_size)
            for batch in range(batches):
                indices = np.arange(batch * batch_size, (batch + 1) * batch_size, 1)
                for i in indices:
                    X_i, Y_i = valid_X[i, :], valid_Y[i]
                    Y_hat_i = np.dot(X_i, beta)
                    L_i = Y_i - Y_hat_i
                    epoch_loss += 0.5 * L_i * L_i
            epoch_loss /= valid_size
            validation_loss_history.append(epoch_loss)
            print 'Validation loss after epoch %d = %f' % ((epoch + 1), epoch_loss)
            if np.argmin(validation_loss_history) == len(validation_loss_history) - 1:
                print 'Best model so far, Saving it to disk'
                np.save('models/best.npy', beta)
            if np.argmin(validation_loss_history) < len(validation_loss_history) - patience:
                print 'Stopping training'

def test(test_data, model):
    model = np.load(model)
    test_X, test_Y = test_data
    test_size = test_X.shape[0]
    epoch_loss = 0.0
    batch_size = 1
    batches = int(float(test_size) / batch_size)
    for batch in range(batches):
        indices = np.arange(batch * batch_size, (batch + 1) * batch_size, 1)
        for i in indices:
            X_i, Y_i = test_X[i, :], test_Y[i]
            Y_hat_i = np.dot(X_i, model)
            L_i = Y_i - Y_hat_i
            epoch_loss += 0.5 * L_i * L_i
    epoch_loss /= test_size
    print 'Test loss = %f' % epoch_loss

def predict(test_data, model):
    model = np.load(model)
    test_X, test_Y = test_data
    test_size = test_X.shape[0]
    Y_hat = []
    for i in range(test_size):
        X_i, Y_i = test_X[i, :], test_Y[i]
        Y_hat_i = np.dot(X_i, model)
        Y_hat.append(Y_hat_i)
    return Y_hat

def plot(Y, Y_hat):
    plt.scatter(Y, Y_hat)

if __name__ == '__main__':
    X, Y = loadData('./data/20_3.txt')
    features = getFeatures(X, 1)
    data = [features, Y]
    train_data, valid_data, test_data = partitionData(data)
    train(train_data, valid_data) 
    test(test_data, 'models/best.npy')
    Y = test_data[1]
    Y_hat = predict(test_data, 'models/best.npy')
    plot(Y, Y_hat)

