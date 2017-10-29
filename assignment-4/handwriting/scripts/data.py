import numpy as np

labels = ['bA', 'dA', 'lA']
label_feats = {}
for label in labels:
    raw = [line.strip() for line in open('data/raw/%s.ldf' % label, 'r').readlines()]
    features = []
    for line in raw:
        if line == '1' or line == label:
            continue
        feats_ = np.array([float(word) for word in line.split()][1 : ])
        features.append(feats_)
    features = np.array(features)
    label_feats[label] = features

for label in labels:
    print label_feats[label].shape
