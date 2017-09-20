import numpy as np

def preprocessData(X):
    X_mean = np.mean(X, axis = 0)
    X = X - X_mean
    X_square = X * X
    var = np.sum(X_square, axis = 0) / (X.shape[0] - 1)
    sig = np.sqrt(var)
    X = X / sig
    return X
 
if __name__ == '__main__':
    lines = [line.strip().split(' ') for line in open('data/20_real.txt', 'r')]
    data = np.array([[float(one), float(two)] for one, two in lines])
    data = preprocessData(data) 
    print data.shape
    print data[0]
    
    f = open('data/20_real_p.txt', 'w')
    for x, y in data:
        f.write(str(x) + ' ' + str(y) + '\n')
    f.close()
