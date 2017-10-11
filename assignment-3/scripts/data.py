import h5py
import os
import numpy as np

def getFeatures(image_path):
    return np.array([line.strip().split() for line in open(image_path, 'r').readlines()], dtype = np.float32)

# Read features
class_dict = {}# Image-path mapping
for class_folder in os.listdir('data/features/'):
    class_dict[class_folder] = []
    class_path = os.path.join('data/features', class_folder)
    for image in os.listdir(class_path):
        image_path = os.path.join(class_path, image)
        class_dict[class_folder].append(getFeatures(image_path))
    print class_folder, len(class_dict[class_folder])
    np.random.shuffle(class_dict[class_folder])
print 'Read data from disk'

def preprocessData(X):
    X_mean = np.mean(X, axis = 0)
    X = X - X_mean
    X_square = X * X
    var = np.sum(X_square, axis = 0) / (X.shape[0] - 1)
    sig = np.sqrt(var)
    return X_mean, sig

# Pool all data
keys = class_dict.keys()
all_data = []
for key in keys:
    all_data += class_dict[key]
all_data = np.array(all_data, dtype = np.float32)

# Get mean and sig for features
mean, sig = preprocessData(all_data)

# Normalize
for key in keys:
    for i in range(len(class_dict[key])):
        class_dict[key][i] -= mean
        class_dict[key][i] /= sig
print 'Features normalization done'

# Partition data 
split = [0.7, 0.15, 0.15]
data = ({}, {}, {}) # Train, val, test : each contains a dict of indexed by class-names
for class_name in class_dict.keys():
    size = len(class_dict[class_name])
    lower = 0
    # Train, val, test
    for i in [0, 1, 2]:
        upper = lower + int(size * split[i])
        data[i][class_name] = class_dict[class_name][lower : upper]
        lower = upper
print 'Partitioned data'

# Crete train, valid and test objects 
data_file_map = ['train', 'valid', 'test']
class_name_map = {'forest' : 0, 'street' : 1, 'highway' : 2}
for stage in [0, 1, 2]:
    print '######################################'
    dataset = h5py.File('data/%s.h5' % data_file_map[stage], 'w')
    for class_name, class_index in class_name_map.iteritems():
        points = np.array(data[stage][class_name], dtype = np.float32)
        print 'Writing %s to %s' % (class_name, data_file_map[stage])
        print 'Shape of written object is', points.shape
        dataset.create_dataset(name = str(class_index), dtype = points.dtype, shape = points.shape, data = points)
    print '######################################'
    dataset.close()

