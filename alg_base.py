import numpy as np 
from random import random, choices
import copy
from patterns import evaluate, neighbors, pairs
from targets import getGoal 

def evalf(param, **xt):
    if callable(param):
        return param(**xt)
    return param

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
        self.timer = None
    def initAttributes(self, **args):
        self.__dict__.update(args)
    def addProcedure(self, proc):
        self.additionalProcedures.append(proc)
    def start(self, envAttrs, indAttrs, shadows = ""):
        if self.timer != None:
            self.timer.startGlobal()
        self.goal = getGoal(self.goal)
        # environment
        keys = ["target", "goal", "time", "timer"] + envAttrs.split()
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
    def newGeneration(self):
        self.env['time'] += 1
        for proc in self.additionalProcedures:
            proc(self.population, self.env)
    def finish(self):
        if self.timer != None:
            self.timer.stopGlobal()
# Common stuff

class fillAttribute:
    def __init__(self, op):
        self.op = op
    def __call__(self, ind, **xt):
        key = xt['key']
        if callable(self.op):
            ind[key] = self.op(xt)
        elif np.isscalar(self.op):
            ind[key] = self.op
        else:
            ind[key] = self.op.copy()

def copyAttribute(ind, **xt):
    keyFrom = xt['keyFrom']
    keyTo = xt['keyTo']
    if np.isscalar(ind[keyFrom]):
        ind[keyTo] = ind[keyFrom]
    else:
        ind[keyTo] = ind[keyFrom].copy()

class mixture:
    def __init__(self, methods, probs):
        self.methods = methods + [None]
        self.probs = probs + [1 - np.sum(probs)]
    def __call__(self, inds, **xt):
        m = choices(self.methods, weights=self.probs)[0]
        if m:
            m(inds, **xt)

class probOp:
    def __init__(self, method, prob):
        self.method = method
        self.prob = prob
    def __call__(self, inds, **xt):
        prob = evalf(self.prob, inds=inds, **xt)
        if random() < prob:
            self.method(inds, **xt)

class timedOp:
    def __init__(self, method, dt):
        self.method = method
        self.dt = dt
    def __call__(self, inds, **xt):
        t = xt['time']
        if t % self.dt == 0:
            self.method(inds, **xt)

class shuffled:
    def __init__(self, op):
        self.op = op
    def __call__(self, population, **xt):
        popSize = len(population)
        P = list(range(popSize))
        np.random.shuffle(P)
        neighbors(population, self.op, P, **xt)

class selected:
    def __init__(self, op):
        self.op = op
    def __call__(self, population, **xt):
        shadow = []
        for i in range(len(population)):
            j = samplex(len(population), 1, [i])[0]
            shadow.append(copy.deepcopy(population[j]))
        pairs(population, shadow, self.op, **xt)

def samplex(n, m, x):
    s = list(set(range(n)) - set(x))
    return list(np.random.choice(s, m, False))

def simpleMove(ind, **xt):
    keyx = xt['keyx']
    keyv = xt['keyv']
    dt = xt['dt']
    ind[keyx] += dt * ind[keyv]

