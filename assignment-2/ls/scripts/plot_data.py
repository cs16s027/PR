import numpy as np
import h5py
from matplotlib import pyplot as plt

def plotData(data, path, title):
    colors = {'0' : 'blue', '1' : 'green', '2' : 'red'}
    for label in ['0', '1', '2']:
        points = data[label][:]
        print 'Number of points in class %s = %s' % (label, str(points.shape[0]))
        x = points[:, 0]
        y = points[:, 1]
        plt.scatter(x, y, color = colors[label], label = label)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.savefig(path)

if __name__ == '__main__':
    data = h5py.File('data/train.h5', 'r')
    path = 'plots/training_data.png'
    title = 'Training-Data'
    plotData(data, path, title)

