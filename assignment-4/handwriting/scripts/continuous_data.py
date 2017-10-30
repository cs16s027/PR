from scipy.io import loadmat
import matplotlib.pyplot as plt

for f in ['1', '2', '3']:

    data_ = loadmat('data/cont/%s.mat' % f)
    x = data_['xData'][0]
    y = data_['yData'][0]

    for i in range(x.shape[0]):
        plt.scatter(x[i], y[i])

    plt.savefig('plots/viz/%s.jpg' % f)
    plt.clf()

