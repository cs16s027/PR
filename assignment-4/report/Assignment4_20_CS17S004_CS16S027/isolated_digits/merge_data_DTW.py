import numpy as np

labels = ['1', '5', 'z']
data = []

for label in labels:
    label_data = np.load('data/proc/train/%s.npy' % label)
    print len(label_data)
    for d in label_data:
        data.append(d)

data = np.array(data)
print data.shape
np.save('data/proc/train/dtw.npy', data)
