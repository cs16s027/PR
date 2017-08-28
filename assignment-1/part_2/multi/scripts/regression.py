import itertools
import numpy as np


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

if __name__ == '__main__':
    X, Y = loadData('./data/20_3.txt')
    features = getFeatures(X, 2)
    data = [features, Y]
    train_data, valid_data, test_data = partitionData(data)
    
