import os

states = 5
symbols = 10
error = 0.01
for label in ['1', '5', 'z']:
    os.system('./scripts/hmm/train_hmm data/proc/train/%s.seq 0 %s %s %s' % (str(label), str(states), str(symbols), str(error)))
    os.system('mv data/proc/train/%s.seq.hmm models/' % str(label))

