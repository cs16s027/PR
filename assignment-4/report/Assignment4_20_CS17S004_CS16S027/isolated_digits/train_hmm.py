import os
import sys

states = int(sys.argv[2])
symbols = int(sys.argv[1])
error = 0.01
for label in ['1', '5', 'z', 'background']:
    os.system('./scripts/hmm/train_hmm data/proc/train/%s_%s.seq 0 %s %s %s' % (str(label), str(symbols), str(states), str(symbols), str(error)))
    os.system('mv data/proc/train/%s_%s.seq.hmm models/hmm/%s_%s_%s.hmm' % (str(label), str(symbols), str(label), str(symbols), str(states)))

