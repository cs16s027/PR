import itertools

labels = ['1', '5', 'z']
for length in [2, 3, 4, 5]:
    
    hmm_strings = [p for p in itertools.product(labels, repeat = length)]

    for string in hmm_strings:
        hmms = []
        states = 0
        symbols = 0
        for index, char in enumerate(string):
            hmm = [line.strip() for line in open('models/%s.hmm' % char).readlines()]
            hmms += hmm[2 : -4]
            if index + 1 != length: 
                for e in [-4, -3]:
                    hmm_ = hmm[e].split('\t')
                    hmm_[0] = '0.500000'
                    size = len(hmm_[1 : ])
                    hmm[e] = '\t'.join(hmm_)
                    hmms.append(hmm[e])
                #hmms.append('')
                #hmm_ = '0.000000\t' + '\t'.join(['0.000000'] * size)  #'\t'.join([str(1 / float(size)) for i in range(size)])
                #hmms.append(hmm_)
                #hmm_ = '1.000000\t' + '\t'.join(['0.000000'] * size)  #'\t'.join([str(1 / float(size)) for i in range(size)])
                #hmms.append(hmm_)
                hmms.append('')
            else:
                hmms += hmm[-4 : ]
            states += int(hmm[0].split('states: ')[1])
        #states += index
        symbols = int(hmm[1].split('symbols: ')[1])
        hmms = ['states: %s' % states, 'symbols: %s' % symbols] + hmms
        hmm_string_file = open('models/%s.hmm' % ''.join(string), 'w')
        for line in hmms:
            hmm_string_file.write(line + '\n')
        hmm_string_file.close()
        print 'Written model %s' % ''.join(string)

