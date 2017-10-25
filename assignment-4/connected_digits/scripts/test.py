import os
import numpy as np

for data in ['1', '5', 'z']:
    for model in ['1', '5', 'z']:
        os.system('./scripts/hmm/test_hmm data/proc/test/%s.seq models/%s.seq.hmm | grep "alpha for" | cut -d"=" -f2 | cut -d" " -f2 > results/%s_%s.pred' % (data, model, model, data))
    os.system('paste -d" " results/1_%s.pred results/5_%s.pred results/z_%s.pred > results/%s.pred' % (data, data, data, data))

correct = 0.0
total = 0.0
label_map = ['1', '5', 'z']
for label in ['1', '5', 'z']:
    preds = [line.strip().split() for line in open('results/%s.pred' % label, 'r').readlines()]
    for point in preds:
        pred_label = label_map[np.argmax(np.float32(point))]
        if pred_label == label: 
            correct += 1
        total += 1
print correct / total * 100 
