import numpy as np 
import functools as ft
from metrics import timing

@timing
def evaluate(population, **xt): # this function can be parallelized by MPI for heavy target functions
    target = xt['target'] 
    keyx = xt['keyx'] 
    keyf = xt['keyf'] 
    for ind in population:
        reEval = 'reEval' not in xt or ind[xt['reEval']]
        ind[keyf] = target(ind[keyx], ind[keyf], reEval) 

@timing
def foreach(population, op, **xt):
    for ind in population: # parallel loop
        op(ind, **xt)

@timing
def neighbors(population, op, P, **xt):
    for i in range(len(P) // 2): # parallel loop
        ind1, ind2 = population[P[2 * i]], population[P[2 * i + 1]]
        op((ind1, ind2), twoway=True, **xt) 

@timing
def pairs(population1, population2, op, **xt):
    for ind1, ind2 in zip(population1, population2): # parallel loop
        op((ind1, ind2), twoway=False, **xt) 

@timing
def pop2ind(population1, population2, op, **xt):
    for i in range(len(population1)): # parallel loop
        ind = population1[i]
        op(ind, population2, index=i, **xt) 

# reducing population into single value (can be parallelized as binary tree)
@timing
def reducePop(population, extract, op, post, **xt):
    if 'initVal' in xt: #!= None:
        return post(ft.reduce(op, map(lambda ind: extract(ind), population), xt['initVal']))
    return post(ft.reduce(op, map(lambda ind: extract(ind), population)))

@timing
def signals(population, metrics, shape, reduce, **xt):
    keyx = xt['keyx']
    keys = xt['keys']
    n = len(population)
    D = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i+1, n):
            D[i, j] = D[j, i] = metrics(population[i][keyx], population[j][keyx])
    for i in range(n):
        ind = population[i]
        S = np.zeros(n)
        for j in range(n):
            S[j] = shape(D[i][j], inds=[population[i], population[j]], **xt) 
        ind[keys] = reduce(S)
