# functions for parallel execution

import numpy as np 
from functools import reduce

def evaluate(population, args):
    metrics = args['metrics']
    keyx = args['keyx']
    keyf = args['keyf']
    for ind in population:
        if 'reEval' in args:
            reEval = ind[args['reEval']]
        else:
            reEval = True
        metrics.newEval(ind, keyx, keyf, reEval)

def foreach(population, op, args):
    for ind in population:
        op(ind, args)

def randomPairwise(population, op, P, args):
    for i in range(len(P) // 2):
        ind1, ind2 = population[P[2 * i]], population[P[2 * i + 1]]
        op((ind1, ind2), args)

def pairwise(population1, population2, op, args):
    for ind1, ind2 in zip(population1, population2):
        op((ind1, ind2), args)

def pop2ind(population1, population2, op, args):
    for i in range(len(population1)):
        ind = population1[i]
        a = {'index':i}
        a.update(args)
        op(ind, population2, a)

def reducePop(population, extract, op, post, initVal=None):
    if initVal != None:
        return post(reduce(op, map(lambda ind: extract(ind), population), initVal))
    return post(reduce(op, map(lambda ind: extract(ind), population)))
