import h5py

def y(x, w):
    y_ = np.dot(w, x)
    if y_ > 0:
        return 1
    return -1

train = h5py.File('data/train.h5', 'r')
print train['1'][:].shape

labels = ['0', '1', '2']
for label in labels:
    positive = label
    negative = list(set(labels) - set(label))
    positive = train[positive][:]

