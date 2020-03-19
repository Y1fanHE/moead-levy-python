import numpy as np

def tchebycheff(f, w, z):
    """
    Compute the Tchebycheff cost

    parameter
    ----------
    f: 1D-Array
      fitness values of an individual
    w: 1D-Array
      weight vector
    z: reference point

    return
    ----------
    float
      Tchebycheff cost value
    """
    return np.max( w * np.abs(f - z) )