import numpy as np
import matplotlib.pyplot as plt

for label in ['bA', 'dA', 'lA']:
    data = np.load('data/proc/train/%s_x,y.npy' % label)
    index = 0
    for seq in data:
        index += 1
        plt.scatter(seq[:, 0], seq[:, 1])
        plt.savefig('plots/viz/train/%s_%s.jpg' % (label, index))
        plt.clf()
