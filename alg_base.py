import numpy as np 
from random import random, choices
import copy
from patterns import evaluate, neighbors, pairs
from targets import getGoal 

def evalf(x, args):
    if callable(x):
        return x(args)
    return x

class algorithm: 
    def __init__(self):
        self.target = None
        self.goal = "min"
        self.stop = lambda x: None
        self.popSize = None
        self.opInit = None
        self.env = None
        self.population = None
        self.additionalProcedures = []
    def initAttributes(self, args):
        self.__dict__.update(args)
    def addProcedure(self, proc):
        self.additionalProcedures.append(proc)
    def start(self, envAttrs, indAttrs, shadows = ""):
        self.goal = getGoal(self.goal)
        # environment
        keys = ["target", "goal", "time"] + envAttrs.split()
        self.env = dict(zip(keys, [None] * len(keys)))
        for key in keys:
            if key in self.__dict__:
                self.env[key] = self.__dict__[key]
        self.env['time'] = 0
        # population
        keys = indAttrs.split()
        ind = dict(zip(keys, [None] * len(keys)))
        self.population = [ind.copy() for i in range(self.popSize)]
        if self.opInit == None:
            self.opInit = fillAttribute(self.target.defaultInit())
        # shadow populations
        for sh in shadows.split():
            self.__dict__[sh] = [ind.copy() for i in range(self.popSize)]
    def args(self, **a):
        a.update({'env':self.env})
        return a
    def evaluateAll(self, keyx='x', keyf='f'):
        evaluate(self.population, self.args(keyx=keyx, keyf=keyf))
    def newGeneration(self):
        self.env['time'] += 1
        for proc in self.additionalProcedures:
            proc(self.population, self.env)

# Common stuff

class fillAttribute:
    def __init__(self, op):
        self.op = op
    def __call__(self, ind, args):
        key = args['key']
        if callable(self.op):
            ind[key] = self.op(args)
        elif np.isscalar(self.op):
            ind[key] = self.op
        else:
            ind[key] = self.op.copy()

def copyAttribute(ind, args):
    keyFrom = args['keyFrom']
    keyTo = args['keyTo']
    if np.isscalar(ind[keyFrom]):
        ind[keyTo] = ind[keyFrom]
    else:
        ind[keyTo] = ind[keyFrom].copy()

class mixture:
    def __init__(self, methods, probs):
        self.methods = methods + [None]
        self.probs = probs + [1 - np.sum(probs)]
    def __call__(self, inds, args):
        m = choices(self.methods, weights=self.probs)[0]
        if m:
            m(inds, args)

class probOp:
    def __init__(self, method, prob):
        self.method = method
        self.prob = prob
    def __call__(self, inds, args):
        prob = evalf(self.prob, [args] + list(inds))
        if random() < prob:
            self.method(inds, args)

class timedOp:
    def __init__(self, method, dt):
        self.method = method
        self.dt = dt
    def __call__(self, inds, args):
        t = args['env']['time']
        if t % self.dt == 0:
            self.method(inds, args)


class shuffled:
    def __init__(self, op):
        self.op = op
    def __call__(self, population, args):
        popSize = len(population)
        P = list(range(popSize))
        np.random.shuffle(P)
        neighbors(population, self.op, P, args)

class selected:
    def __init__(self, op):
        self.op = op
    def __call__(self, population, args):
        shadow = []
        for i in range(len(population)):
            j = samplex(len(population), 1, [i])[0]
            shadow.append(copy.deepcopy(population[j]))
        pairs(population, shadow, self.op, args)

def samplex(n, m, x):
    s = list(set(range(n)) - set(x))
    return list(np.random.choice(s, m, False))

def simpleMove(ind, args):
    keyx = args['x']
    keyv = args['v']
    dt = args['dt']
    ind[keyx] += dt * ind[keyv]

