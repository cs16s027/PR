import numpy as np

def softmax(vec):
    m = np.max(vec)
    s = np.sum(np.exp(vec - m))
    vec_ = []
    for v in vec:
        vec_.append(np.exp(v - m) / s)
    return vec_

liks = [line.strip().split() for line in open('pr.txt', 'r')]
f = open('parzen.txt', 'w')
for i in np.arange(0, 131, 1):
    lik = [float(word) for word in liks[i]]
    probs = softmax(lik)
    if i <= 48:
        label = 0
    elif i <= 92:
        label = 1
    else:
        label = 2
    f.write(str(label) + ' ' + ' '.join([str(word) for word in probs]) + '\n')
f.close()

