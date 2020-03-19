import numpy as np

def SCH(x):
    f1 = x[0] ** 2
    f2 = (x[0] - 2) ** 2
    return np.array([f1, f2])