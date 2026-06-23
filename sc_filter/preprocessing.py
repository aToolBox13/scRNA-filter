import numpy as np
from scipy.sparse import issparse

def normalize_counts(count_matrix, scale_factor=10000.0):
    if issparse(count_matrix):
        matrix = count_matrix.toarray()
    else:
        matrix = np.array(count_matrix, dtype=np.float32)
        
    cell_sums = matrix.sum(axis=1, keepdims=True)
    cell_sums[cell_sums == 0] = 1.0
    
    normalized = np.log1p((matrix / cell_sums) * scale_factor)
    return normalized
