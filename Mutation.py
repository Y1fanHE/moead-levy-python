import random
import numpy as np
from math import gamma as G

def poly_mutation(x, eta_m, xl, xu):
    """
    Implement polynomial mutation

    parameter
    ----------
    x: 1D-Array
      mutant vector (individual)
    eta_m: float
      index parameter of polynomial mutation
    xl, xu: float
      lower and upper boundary of decision variables

    return
    ----------
    1D-Array
      an offspring
    """
    n_var = len(x)
    is_mutated = np.random.choice([0, 1], n_var, p=[1 - 1 / n_var, 1 / n_var])
    mu = random.random() 
    if mu < 0.5:
        sigmaq = (2 * mu) ** ( 1 / (eta_m + 1) ) - 1
    else:
        sigmaq = 1 - ( 2 * (1 - mu) ) ** ( 1 / (eta_m + 1) )
    return x + is_mutated * sigmaq * (xu - xl)

def lf_mutation(xi, xj, alpha, beta):
    """
    Implement levy flight mutation

    parameter
    ----------
    xi: 1D-Array
      current individual
    xj: 1D-Array
      random individual
    alpha: float
      positive scaling factor
    beta: float
      stability parameter (0, 2)

    return
    ----------
    1D-Array
      an offspring
    """
    n_var = len(xi)
    xi_ = xi + levy(alpha, beta, n_var) * (xi - xj)
    return xi_

def levy(alpha, beta, n_var):
    if beta < 0:
        print("Error: Stable distribution requires a beta between 0.3 to 1.99.")
        return None
    elif beta < 0.3:
        return gutowski(alpha, beta, n_var)
    elif beta <= 1.99:
        return mantegna(alpha, beta, n_var)
    else:
        print("Error: Stable distribution requires a beta between 0.3 to 1.99.")
        return None

def mantegna(alpha, beta, n_var):
    if beta < 0.3 or beta > 1.99:
        print("Error: Mantegna's algorithm requires a beta between 0.3 to 1.99.")
        return None
    num = G(1 + beta) * np.sin(np.pi * beta / 2)
    den = G( (1 + beta) / 2 ) * beta * 2 ** ( (beta - 1) / 2 )
    sigma_u = (num / den) ** (1 / beta)
    u = np.random.normal(0, sigma_u, n_var)
    v = np.random.normal(0, 1, n_var)
    l = alpha * u / ( np.abs(v) ** (1 / beta) )
    return l

def gutowski(alpha, beta, n_var):
    if beta <= 0 or beta > 2:
        print("Error: Gutowski's algorithm requires a beta between 0 to 2.")
        return None
    u = np.random.uniform(0, 1, n_var)
    l = u ** (- 1 / beta) - 1
    sgn = np.random.choice([1, -1], n_var)
    l = alpha * sgn * l
    return l

def fix_bound(x, xl, xu):
    x = np.clip(x, xl, xu)
    return x