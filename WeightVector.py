import numpy as np
from scipy.spatial.distance import cdist

def das_dennis(n_part, n_obj):
    """
    Generate a set of weight vectors using SLD method

    parameter
    ----------
    n_part: int
      a user-defined parameter showing the partition number on each objective
    n_obj: int
      number of objectives
    
    return
    ----------
    2D-Array
      a matrix where each row is a weight vector
    """
    if n_part == 0:
        return np.full((1, n_obj), 1 / n_obj)
    else:
        ref_dirs = []
        ref_dir = np.full(n_obj, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_part, n_part, 0)
        return np.concatenate(ref_dirs, axis=0)

def determine_neighbor(ref_dirs, n_neighbors):
    """
    Determine neighbor based on the set of weight vectors

    parameter
    ----------
    ref_dirs: 2D-Array
      a matrix of weight vector where each row is a weight vector
    n_neighbors: int
      number of neighbors
    """
    return np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_neighbors]

def das_dennis_recursion(ref_dirs, ref_dir, n_part, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_part)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_part)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_part, beta - i, depth + 1)