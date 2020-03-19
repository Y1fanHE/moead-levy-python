import numpy as np

def init_ref_point(F):
    """
    Initialize a reference point

    parameter
    ----------
    F: 2D-Array
      a matrix where each row showing fitness values of an individual
    
    return
    ----------
    1D-array
      position of reference point
    """
    return np.min(F, axis=0)

def update_ref_point(ref_point, fy):
    """
    Update the reference point by an offspring

    parameter
    ----------
    ref_point: 1D-Array
      the position of original reference point
    fy: 1D-Array
      the fitness values of the offspring

    return
    ----------
    1D-Array
      the position of the updated reference point
    """
    tmp = np.vstack([ref_point, fy])
    return np.min(tmp, axis=0)