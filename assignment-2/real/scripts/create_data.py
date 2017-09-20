import numpy as np
import h5py

# Read data from file
input_data = [line.strip().split(' ') for line in open('data/20_real_p.txt', 'r').readlines()]

# Dictionary to store data classwise
class_dict = {0 : [], 1 : [], 2 : []}

# Populate the dictionary
for index, point in enumerate(input_data):
    if 0 <= index and index < 500:
        label = 0
    elif 500 <= index and index < 1000:
        label = 1
    else:
        label = 2

    feature = [np.float(word) for word in point]
    class_dict[label].append(feature)

# Set random seed for reproducability
np.random.seed(0)
# Shuffle the data in each class
for label in [0, 1, 2]:
    np.random.shuffle(class_dict[label])

# Partition data and write to disk
data_write = {0 : 'train', 1 : 'valid', 2 : 'test'}
# Specify partition limits for train, valdation and test
split = [(0, 0.7), (0.7, 0.9), (0.9, 1)]

# stages 0, 1 and 2 correspond to train, validation and test respectively
for stage in [0, 1, 2]:
    print 'Writing %s data to disk' % data_write[stage]
    # data_h5 is the object that will be used to write data to h5 file
    data_h5 = h5py.File('data/%s.h5' % data_write[stage], 'w')
    # Run through each label
    for label in [0, 1, 2]:
        # Size of the data points belonging to this label 
        size = len(class_dict[label])
        # Lower and Upper specify the slice indices
        lower, upper = [int(size * word) for word in split[stage]]
        # Extract the data that needs to be written
        data_stage_label = np.array(class_dict[label][lower : upper], dtype = np.float32)
        # Write data to hdf5 file
        data_h5.create_dataset(name = str(label), shape = data_stage_label.shape, dtype = data_stage_label.dtype, data = data_stage_label)
        print 'Size of class %s = %s' % (str(label), str(data_stage_label.shape))
    data_h5.close()
        
