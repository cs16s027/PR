import numpy as np
import h5py
from matplotlib import pyplot as plt

def plotData(data, path, title, figure = 0):
    print 'Plotting data'
    if 'Test' in title:
        colors = {'0' : 'black', '1' : 'black', '2' : 'black'}
    else:
        colors = {'0' : 'blue', '1' : 'green', '2' : 'red'}
    fig = plt.figure(figure)
    ax = fig.gca()

    for label in ['0', '1', '2']:
        points = data[label][:]
        #print 'Number of points in class %s = %s' % (label, str(points.shape[0]))
        x = points[:, 0]
        y = points[:, 1]
        ax.scatter(x, y, color = colors[label], label = label)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    # Name the classes
    ax.text(6, 0, 'Class-0')
    ax.text(-6, 9.0, 'Class-2')
    ax.text(13, 6.5, 'Class-1')
    plt.title(title)
    plt.savefig(path)

if __name__ == '__main__':
    data = h5py.File('data/train.h5', 'r')
    path = 'plots/training_data.jpg'
    plotData(data, path, 'Test plot', figure = 0)

