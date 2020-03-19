import os
import argparse
import yaml
import random
import numpy as np
from scipy.special import comb

from Factory import set_problem
from WeightVector import das_dennis, determine_neighbor
from Population import init_pop, eval_pop
from ReferencePoint import init_ref_point, update_ref_point
from Mutation import lf_mutation, poly_mutation, fix_bound
from Decomposition import tchebycheff


###################
# parse arguments #
###################

parser = argparse.ArgumentParser()
parser.add_argument('params', type=argparse.FileType('r'))
parser.add_argument('seed', type=int)
args = parser.parse_args()

##################
# set parameters #
##################

params = yaml.safe_load(args.params)                                # read config file

random.seed(args.seed)
np.random.seed(args.seed)

prefix = params['prefix']                                           # set prefix of record in this run

prob_name = params['prob_name']                                     # set optimization problem
f = set_problem(prob_name)
n_obj = params['n_obj']                                             # set number of objectives
n_var = params['n_var']                                             # set number of variables
xl = params['xl']                                                   # set boundary of variables
xu = params['xu']

n_part = params['n_part']                                           # set number of partitions

n_eval = params['n_eval']                                           # set maximum number of evaluation
n_pop = int( comb(n_obj + n_part - 1, n_obj - 1) )                  # compute population size
n_gen = n_eval // n_pop                                             # compute maximum generation

n_neb = params['n_neb']                                             # set neighbor size
sigma = params['sigma']                                             # set probability to select parent from neighbor
nr = params['nr']                                                   # set maximum update counts for one offspring

alpha = params['alpha']                                             # set scaling factor of levy flight mutation
beta = params['beta']                                               # set stability parameter of levy flight mutation
etam = params['etam']                                               # set index parameter of polynomial mutation

#################
# start program #
#################

os.makedirs(f'./{prefix}', exist_ok=True)                           # create a folder to include running results
os.makedirs(f'./{prefix}/history/', exist_ok=True)
os.makedirs(f'./{prefix}/history/{args.seed}', exist_ok=True)

W = das_dennis(n_part, n_obj)                                       # generate a set of weight vectors
B = determine_neighbor(W, n_neb)                                    # determine neighbor
X = init_pop(n_pop, n_var, xl, xu)                                  # initialize a population
F = eval_pop(X, f)                                                  # evaluate fitness
z = init_ref_point(F)                                               # determine a reference point

for c_gen in range(1, n_gen):                                       # star main loop

    result = np.hstack([F, X])                                      # record objective values and decision variables
    np.savetxt(f'./{prefix}/history/{args.seed}/{c_gen}.csv', result)

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

result = np.hstack([F, X])                                          # record objective values and decision variables
np.savetxt(f'./{prefix}/history/{args.seed}/{n_gen}.csv', result)
np.savetxt(f'./{prefix}/{args.seed}_final.csv', result)