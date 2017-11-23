import h5py
import os
import numpy as np

def getFeatures(image_path):
    return np.array([line.strip().split() for line in open(image_path, 'r').readlines()][1:], dtype = np.float32)

# Read features
class_dict = {}# Image-path mapping
for class_folder in os.listdir('data/raw/'):
    class_dict[class_folder] = []
    class_path = os.path.join('data/raw', class_folder)
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

# Partition data 
split = [0.7, 0.3]
data = ({}, {}) # Train, test : each contains a dict of indexed by class-names
for class_name in class_dict.keys():
    size = len(class_dict[class_name])
    lower = 0
    # Train, test
    for i in [0, 1]:
        upper = lower + int(size * split[i])
        data[i][class_name] = class_dict[class_name][lower : upper]
        lower = upper
print 'Partitioned data'

# Get mean and covariance
keys = class_dict.keys()
all_data = []
for key in keys:
    all_data += data[0][key]
all_data = np.concatenate(all_data)
all_data = np.reshape(all_data, (-1, 38))
all_data = np.array(all_data, dtype = np.float32)

# Get mean and sig for features
mean, sig = preprocessData(all_data)

# Normalize
for stage  in [0, 1]:
    for label in keys:
        for seq in range(len(data[stage][label])):
            for vec in range(data[stage][label][seq].shape[0]): 
                data[stage][label][seq][vec] -= mean
                data[stage][label][seq][vec] /= sig
print 'Features normalization done'

# Crete train, valid and test objects 
data_file_map = ['train', 'test']
class_name_map = {'1' : 0, '5' : 1, 'z' : 2}
for stage in [0, 1]:
    for label in keys:
        np.save('data/proc/%s/%s.npy' % (data_file_map[stage], label), data[stage][label])
        print stage, label, len(data[stage][label])
