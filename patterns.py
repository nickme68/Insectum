import numpy as np 
import functools as ft

def evaluate(population, **xt): # this function can be parallelized by MPI for heavy target functions
    target = xt['target'] 
    keyx = xt['keyx'] 
    keyf = xt['keyf'] 
    for ind in population:
        reEval = 'reEval' not in xt or ind[xt['reEval']]
        ind[keyf] = target(ind[keyx], ind[keyf], reEval) 

def foreach(population, op, **xt):
    for ind in population: # parallel loop
        op(ind, **xt)

def neighbors(population, op, P, **xt):
    for i in range(len(P) // 2): # parallel loop
        ind1, ind2 = population[P[2 * i]], population[P[2 * i + 1]]
        op((ind1, ind2), twoway=True, **xt) 

def pairs(population1, population2, op, **xt):
    for ind1, ind2 in zip(population1, population2): # parallel loop
        op((ind1, ind2), twoway=False, **xt) 

def pop2ind(population1, population2, op, **xt):
    for i in range(len(population1)): # parallel loop
        ind = population1[i]
        op(ind, population2, index=i, **xt) 

# reducing population into single value (can be parallelized as binary tree)
def reducePop(population, extract, op, post, initVal=None):
    if initVal != None:
        return post(ft.reduce(op, map(lambda ind: extract(ind), population), initVal))
    return post(ft.reduce(op, map(lambda ind: extract(ind), population)))
