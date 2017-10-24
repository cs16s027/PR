import os

for model in ['1', '5', 'z']:
    for data in ['1', '5', 'z']:
        os.system('./scripts/hmm/test_hmm data/proc/test/%s.seq models/%s.seq.hmm | grep "alpha for" | cut -d"=" -f2 | cut -d" " -f2 > results/%s_%s.pred' % (data, model, model, data))

for label in ['1', '5', 'z']:
    print 'Ground truth is ', label
    files = [open('results/%s_%s.pred' % (model, label), 'r') for model in ['1', '5', 'z']]
    for f in files:
        [line.strip() for line in f.readlines()]
