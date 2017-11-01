import os
import sys

_, feat_indicator, symbols, states = sys.argv
states = int(states)
symbols = int(symbols)
error = 0.01
for label in ['bA', 'dA', 'lA', 'background']:
    os.system('./scripts/hmm/train_hmm data/proc/train/%s_%s_%s.seq 0 %s %s %s' % (str(label), feat_indicator, str(symbols), str(states), str(symbols), str(error)))
    os.system('mv data/proc/train/%s_%s_%s.seq.hmm models/hmm/%s_%s_%s_%s.hmm' % (str(label), feat_indicator, str(symbols), str(label), feat_indicator, str(symbols), str(states)))

