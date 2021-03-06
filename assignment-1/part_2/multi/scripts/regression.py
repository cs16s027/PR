import sys
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
        features_i = [1.0]
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
def train(train_data, valid_data, epochs, final_model, ridge, finetune = False, initial_model = False):
    # Data
    train_X, train_Y = train_data
    train_size = train_X.shape[0]
    print '\n### Data ###'
    print 'Training on %d data-points' % train_size
    valid_X, valid_Y = valid_data
    valid_size = valid_X.shape[0]
    print 'Validating on %d data-points' % valid_size

    # Variable initialization
    print '\n### Variable Initialization ###'
    if finetune == True:
        beta = np.load(initial_model)
        print 'Pre-trained params'
    else:
        params_dim = train_X.shape[1]    
        print 'Number of parameters = %d' % params_dim
        beta = np.zeros(params_dim)
        print 'Zero initialization'

    # Hyperparameters
    batch_size = 16
    lr = 0.001
    patience = 5
    print '\n### Hyperparameters ###'
    print 'Training for %d epochs with a batch size of %d and learning rate of %f' % (epochs, batch_size, lr)
    print 'Early stopping has been enabled with patience set to %d' % patience
    
    # Training
    print '\n### Training and Validation ###'
    training_loss_history = []
    validation_loss_history = []
    for epoch in range(epochs):
        batches = int(np.ceil(float(train_size) / batch_size))
        for batch in range(batches):
            low_index, high_index = batch * batch_size, (batch + 1) * batch_size
            if high_index >= train_size:
                high_index = train_size
            indices = np.arange(low_index, high_index, 1)
            batch_loss = 0.0
            batch_loss_grad = np.zeros_like(beta)
            for i in indices:
                X_i, Y_i = train_X[i, :], train_Y[i]
                Y_hat_i = np.dot(X_i, beta)
                L_i = Y_i - Y_hat_i
                batch_loss_grad += L_i * X_i 
            batch_loss_grad -= ridge * np.append(0, beta[1 : ])
            beta += lr * batch_loss_grad
        # Training-Validation-Loss Calculation
        if (epoch + 1) % 1 == 0:
            train_loss, valid_loss = 0.0, 0.0
            sizes = [train_size, valid_size]
            losses = [train_loss, valid_loss]
            histories = [training_loss_history, validation_loss_history]
            data = [(train_X, train_Y), (valid_X, valid_Y)]
            loss_dict = {0 : 'Training', 1 : 'Validation'}
            for mode in [0, 1]:
                X, Y = data[mode]
                for i in range(sizes[mode]):
                    X_i, Y_i = X[i, :], Y[i]
                    Y_hat_i = np.dot(X_i, beta)
                    L_i = Y_i - Y_hat_i
                    losses[mode] += 0.5 * L_i * L_i 
                losses[mode] /= sizes[mode]
                losses[mode] += 0.5 * ridge * np.dot(beta[1 : ], beta[1 : ])
                histories[mode].append(losses[mode])
                print '%s loss after epoch %d = %f' % (loss_dict[mode], (epoch + 1), losses[mode])
            if np.argmin(validation_loss_history) == len(validation_loss_history) - 1:
                print 'Best model so far, Saving it to disk'
                np.save(final_model, beta)
            if np.argmin(validation_loss_history) < len(validation_loss_history) - patience:
                print 'Stopping training'
                #break

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax.set_yscale('log')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.plot(np.arange(1, epochs + 1, 1), validation_loss_history, color = 'blue', label = 'Valid')
    ax.plot(np.arange(1, epochs + 1, 1), training_loss_history, color = 'red', label = 'Train')
    plt.legend()
    plt.savefig('plots/loss.png')
    plt.clf()

def test(test_data, model, ridge):
    model = np.load(model)
    test_X, test_Y = test_data
    test_size = test_X.shape[0]
    test_loss = 0.0
    for i in range(test_size):
        X_i, Y_i = test_X[i, :], test_Y[i]
        Y_hat_i = np.dot(X_i, model)
        L_i = Y_i - Y_hat_i
        test_loss += 0.5 * L_i * L_i 
    test_loss /= test_size
    test_loss += 0.5 * ridge * np.dot(model[1 : ], model[1 : ])
    print 'Test loss = %f' % test_loss

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

def plot(Y, Y_hat, plotname, epochs):
    plt.scatter(Y, Y_hat)
    Y_min, Y_max = np.min(Y), np.max(Y)
    plt.plot([Y_min, Y_max], [Y_min, Y_max], color = 'black')
    plt.xlabel('Y')
    plt.ylabel('Y_hat')
    plt.title('Y_hat vs Y @ %d epochs' % (epochs))
    plt.savefig(plotname)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage : python regression.py <data> <ridge> <epochs> <final_model> <finetune> <initial_model> <plot>'
        print '<data> : input data'
        print '<ridge> : parameter for ridge term'
        print '<epochs> : number of epochs to train the model'
        print '<final_model> : path to save the final model after training'
        print '<finetune> : Y or N'
        print '<intial_model> : initial model if finetuning'
        print '<plot> : plot of Y_hat vs Y'
        exit()

    #### Argparsing ####
    _, data, ridge, epochs, final_model, finetune, initial_model, plotname = sys.argv
    finetune = False
    if finetune == 'Y':
        finetune = True
    ridge = float(ridge)
    epochs = int(epochs)
    ####################
    X, Y = loadData(data)
    features = getFeatures(X, 1)
    data = [features, Y]
    train_data, valid_data, test_data = partitionData(data)
    train(train_data, valid_data, epochs, final_model, ridge, finetune = finetune, initial_model = initial_model) 
    test(test_data, final_model, ridge)
    Y = test_data[1]
    Y_hat = predict(test_data, final_model)
    plot(Y, Y_hat, plotname, epochs)

