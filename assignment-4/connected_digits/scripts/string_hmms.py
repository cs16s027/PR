import itertools

length = 3

labels = ['1', '5', 'z']

hmm_strings = [p for p in itertools.product(labels, repeat = length)]


for string in hmm_strings:
    hmms = []
    states = 0
    symbols = 0
    for char in string:
        hmm = [line.strip() for line in open('models/%s.seq.hmm' % char).readlines()]
        hmms += hmm[2 : -3]
        hmm_ = hmm[-3].split('\t')
        hmm_[0] = '0.500000'
        size = len(hmm_[1 : ])
        hmm[-3] = '\t'.join(hmm_)
        hmms += hmm[-3]
        hmms += ''
        hmm_ = '0.500000\t' + '\t'.join([str(1 / float(size)) for i in range(size)])
        hmms += hmm_
        hmms += ''
        states += int(hmm[0].split('states: ')[1])
    symbols = int(hmm[1].split('symbols: ')[1])
    hmms = ['state: %s' % states, 'symbols: %s' % symbols] + hmms
    hmms += ['', '']
    hmm_string_file = open('models/%s.hmm' % ''.join(string), 'w')
    for line in hmms:
        hmm_string_file.write(line + '\n')
    hmm_string_file.close()
    print 'Written model %s' % ''.join(string)

