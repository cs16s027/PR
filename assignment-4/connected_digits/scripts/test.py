import os
import numpy as np

datas = os.listdir('data/proc/test1')
models = os.listdir('models')
print 'Truth\tPredicted'
for data in datas:
    data_ = data.split('a.txt')[0]
    predictions = []
    for model in models:
        model_ = model.split('.hmm')[0]
        if len(model_) != len(data_):
            predictions.append(-np.inf)
            continue
        model_file = 'models/%s' % model
        data_file = 'data/proc/test1/%s' % data
        os.system('./scripts/hmm/test_hmm %s %s | grep "alpha for" | cut -d"=" -f2 | cut -d" " -f2 > temp' % (data_file, model_file))
        predictions.append(float(open('temp', 'r').readlines()[0].strip()))
        os.system('rm temp')
    print data.split('a.txt')[0] + '\t' + models[np.argmax(predictions)].split('.hmm')[0]

