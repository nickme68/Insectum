# functions for possible parallel execution

import numpy as np 
from functools import reduce

def evaluate(population, args): # this function can be parallelized by MPI for heavy target functions
    metrics = args['metrics']
    keyx = args['keyx']
    keyf = args['keyf']
    for ind in population:
        reEval = 'reEval' not in args or ind[args['reEval']]
        ind[keyf] = metrics.newEval(ind[keyx], ind[keyf], reEval) 

def foreach(population, op, args):
    for ind in population: # parallel loop
        op(ind, args)

def randomPairwise(population, op, P, args):
    for i in range(len(P) // 2): # parallel loop
        ind1, ind2 = population[P[2 * i]], population[P[2 * i + 1]]
        op((ind1, ind2), args)

def pairwise(population1, population2, op, args):
    for ind1, ind2 in zip(population1, population2): # parallel loop
        op((ind1, ind2), args)

def pop2ind(population1, population2, op, args):
    for i in range(len(population1)): # parallel loop
        ind = population1[i]
        a = {'index':i}
        a.update(args)
        op(ind, population2, a)

# reducing population into single value (can be parallelized as binary tree)
def reducePop(population, extract, op, post, initVal=None):
    if initVal != None:
        return post(reduce(op, map(lambda ind: extract(ind), population), initVal))
    return post(reduce(op, map(lambda ind: extract(ind), population)))
