import numpy as np 
import functools as ft

def addDics(d1, d2):
    a = d1.copy()
    a.update(d2)
    return a

def evaluate(population, args): # this function can be parallelized by MPI for heavy target functions
    target = args['env']['target']
    keyx = args['keyx']
    keyf = args['keyf']
    for ind in population:
        reEval = 'reEval' not in args or ind[args['reEval']]
        ind[keyf] = target(ind[keyx], ind[keyf], reEval) 

def foreach(population, op, args):
    for ind in population: # parallel loop
        op(ind, args)

def neighbors(population, op, P, args):
    for i in range(len(P) // 2): # parallel loop
        ind1, ind2 = population[P[2 * i]], population[P[2 * i + 1]]
        op((ind1, ind2), addDics(args, {'twoway':True}))

def pairs(population1, population2, op, args):
    for ind1, ind2 in zip(population1, population2): # parallel loop
        op((ind1, ind2), addDics(args, {'twoway':False}))

def pop2ind(population1, population2, op, args):
    for i in range(len(population1)): # parallel loop
        ind = population1[i]
        op(ind, population2, addDics(args, {'index':i}))

# reducing population into single value (can be parallelized as binary tree)
def reducePop(population, extract, op, post, initVal=None):
    if initVal != None:
        return post(ft.reduce(op, map(lambda ind: extract(ind), population), initVal))
    return post(ft.reduce(op, map(lambda ind: extract(ind), population)))
