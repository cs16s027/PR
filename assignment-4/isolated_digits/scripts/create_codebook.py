import numpy as np
from sklearn.cluster import KMeans

def cluster(data, K):
    kmeans = KMeans(n_clusters = K, random_state = 0).fit(data)
    return kmeans.cluster_centers_, kmeans.labels_

all_frames = []
for label in ['1', '5', 'z']:
    feats = np.load('data/proc/train/%s.npy' % label)
    all_frames.append(np.concatenate(feats))
all_frames = np.concatenate(all_frames)

codes, code_asign = cluster(all_frames, 10)

np.save('data/proc/codes.npy', codes)