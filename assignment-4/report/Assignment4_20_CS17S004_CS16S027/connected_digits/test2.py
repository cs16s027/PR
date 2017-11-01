import os
import numpy as np

datas = os.listdir('data/proc/test2')
datas = sorted(datas)

models = os.listdir('models')
print 'Truth\tPredicted'
for data in datas:
    predictions = []
    for model in models:
        model_length = model.split('.hmm')[0]
        #if len(model_length) != 3:
        #    predictions.append(-np.inf)
        #    continue
        model_ = model.split('.hmm')[0]
        model_file = 'models/%s' % model
        data_file = 'data/proc/test2/%s' % data
        os.system('./scripts/hmm/test_hmm %s %s | grep "alpha for" | cut -d"=" -f2 | cut -d" " -f2 > temp' % (data_file, model_file))
        predictions.append(float(open('temp', 'r').readlines()[0].strip()))
        os.system('rm temp')
    predictions = np.array(predictions)
    avail = np.array(models)[np.argwhere(predictions == np.amax(predictions))].reshape((-1,))
    avail = sorted(avail)
    print data.split('a.txt')[0] + '\t' + avail[0].split('.hmm')[0]

