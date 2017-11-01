import numpy as np

def dtw(template, test):
    N = template.shape[0]
    M = test.shape[0]
    dtw_matrix = np.ones((N + 1, M + 1), dtype = np.float32) * np.inf
    dtw_matrix[0, 0] = 0.0
    for j in np.arange(1, M + 1, 1):
        for i  in np.arange(1, N + 1, 1):
            cost = np.linalg.norm(template[i - 1] - test[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],\
                                             dtw_matrix[i, j -1],\
                                             dtw_matrix[i -1 , j - 1])
    
    return dtw_matrix[N, M]

