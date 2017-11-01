import numpy as np
import sys
import os
from sklearn.cluster import KMeans

def createCodeBook(feat_indicator, K):
    if os.path.isfile('data/proc/codes_%s_%s.npy' % (feat_indicator, K)):
        return 0
    def cluster(data, K):
        kmeans = KMeans(n_clusters = K, random_state = 0).fit(data)
        return kmeans.cluster_centers_, kmeans.labels_

    all_frames = []
    for label in ['bA', 'dA', 'lA']:
        feats = np.load('data/proc/train/%s_%s.npy' % (label, feat_indicator))
        all_frames.append(np.concatenate(feats))
    all_frames = np.concatenate(all_frames)

    codes, code_asign = cluster(all_frames, K)

    np.save('data/proc/codes_%s_%s.npy' % (feat_indicator, K), codes)
    
if __name__ == '__main__':
    _, feat_indicator, K = sys.argv
    createCodeBook(feat_indicator, int(K))
