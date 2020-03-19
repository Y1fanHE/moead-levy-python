import random
import numpy as np
from scipy.special import comb

from MultiObjectiveProblem import SCH
from WeightVector import das_dennis, determine_neighbor
from Population import init_pop, eval_pop
from ReferencePoint import init_ref_point, update_ref_point
from Mutation import lf_mutation, poly_mutation, fix_bound
from Decomposition import tchebycheff


##################
# set parameters #
##################

f = SCH                                                             # set optimization problem
n_obj = 2                                                           # set number of objectives
n_var = 1                                                           # set number of variables
xl, xu = -100, 100                                                  # set boundary of variables

n_part = 3                                                          # set number of partitions

n_eval = 10000                                                      # set maximum number of evaluation
n_pop = int( comb(n_obj + n_part - 1, n_obj - 1) )                  # compute population size
n_gen = n_eval // n_pop                                             # compute maximum generation

n_neb = n_pop // 2                                                  # set neighbor size
sigma = 0.9                                                         # set probability to select parent from neighbor
nr = 2                                                              # set maximum update counts for one offspring

alpha = 1e-05                                                       # set scaling factor of levy flight mutation
beta = 0.3                                                          # set stability parameter of levy flight mutation
etam = 20                                                           # set index parameter of polynomial mutation

#################
# start program #
#################

W = das_dennis(n_part, n_obj)                                       # generate a set of weight vectors
B = determine_neighbor(W, n_neb)                                    # determine neighbor
X = init_pop(n_pop, n_var, xl, xu)                                  # initialize a population
F = eval_pop(X, f)                                                  # evaluate fitness
z = init_ref_point(F)                                               # determine a reference point

for c_gen in range(n_gen):                                          # star main loop

    for i in np.random.permutation(n_pop):                          # traverse the population

        xi, fi = X[i, :], F[i, :]                                   # get current individual

        if random.random() < sigma:                                 # determine selection pool by probability
            pool = B[i, :]                                          # neighbor as the pool
        else:
            pool = np.arange(n_pop)                                 # population as the pool

        j = np.random.choice(pool)                                  # select a random individual from pool
        xj, fj = X[j, :], F[j, :]

        xi_ = fix_bound( lf_mutation(xi, xj, alpha, beta), xl, xu ) # levy flight mutation
        xi_ = fix_bound( poly_mutation(xi_, etam, xl, xu), xl, xu ) # polynomial mutation
        fi_ = f(xi_)                                                # evaluate offspring

        z = update_ref_point(z, fi_)                                # update reference point

        nc = 0                                                      # initialize the update counter
        for k in np.random.permutation(len(pool)):                  # traverse the selection pool

            fk = F[k, :]                                            # get k-th individual fitness
            wk = W[k, :]                                            # get k-th weight vector

            if tchebycheff(fi_, wk, z) <= tchebycheff(fk, wk, z):   # compare tchebycheff cost of offspring and parent
                X[k] = xi_                                          # update parent
                F[k] = fi_
                nc += 1                                             # cumulate the counter

            if nc >= nr: break                                      # break if counter arrive the upper limit
