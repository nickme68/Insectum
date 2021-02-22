import numpy as np 
import copy
from templates import *
#from tasks import *

def evalf(x, args):
    if callable(x):
        return x(args)
    return x

class algorithm:
    def __init__(self):
        self.metrics = None
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
        # environment
        keys = envAttrs.split()
        self.env = dict(zip(keys, [None] * len(keys)))
        for key in keys:
            if key in self.__dict__:
                self.env[key] = self.__dict__[key]
        # population
        keys = indAttrs.split()
        ind = dict(zip(keys, [None] * len(keys)))
        self.population = [ind.copy() for i in range(self.popSize)]
        if self.opInit == None:
            self.opInit = fillAttribute(self.metrics.task.defaultInit())
        # shadow populations
        for sh in shadows.split():
            self.__dict__[sh] = [ind.copy() for i in range(self.popSize)]
    def args(self, **a):
        a.update({'metrics':self.metrics, 'env':self.env})
        return a
    def evaluateAll(self, keyx='x', keyf='f'):
        evaluate(self.population, self.args(keyx=keyx, keyf=keyf))
    def newGeneration(self):
        self.metrics.newGeneration()
        for proc in self.additionalProcedures:
            proc(self.population, self.env, self.metrics)

# Common stuff

def samplex(n, m, x):
    s = []
    for i in range(n):
        if i not in x:
            s.append(i)
    return list(np.random.choice(s, m, False))

class shuffled:
    def __init__(self, op):
        self.op = op
    def __call__(self, population, args):
        popSize = len(population)
        P = list(range(popSize))
        np.random.shuffle(P)
        randomPairwise(population, self.op, P, args)

class selected:
    def __init__(self, op):
        self.op = op
    def __call__(self, population, args):
        if 'selector' in args['env']:
            selector = args['env']['selector']
        else:
            selector = None
        shadow = []
        for i in range(len(population)):
            if selector: 
                j = selector(i)
            else:
                j = samplex(len(population), 1, [i])[0]
            shadow.append(copy.deepcopy(population[j]))
        a = {'twoway':False}
        a.update(args)
        pairwise(population, shadow, self.op, a)

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

def simpleMove(ind, args):
    keyx = args['x']
    keyv = args['v']
    dt = args['dt']
    ind[keyx] += dt * ind[keyv]

class expCool:
    def __init__(self, x0, q):
        self.x = x0
        self.q = q
        self.gen = 0
    def __call__(self, args):
        gen = args[0]['metrics'].currentGeneration
        if gen == 0:
            return self.x
        if gen > self.gen:
            self.gen = gen
            self.x *= self.q
            # TODO: добавить возможность работы со списками или кортежами
        return self.x

class hypCool:
    def __init__(self, x0, deg):
        self.x0 = x0
        self.x = x0
        self.deg = deg
        self.gen = 0
    def __call__(self, args):
        gen = args[0]['metrics'].currentGeneration
        if gen == 0:
            return self.x
        if gen > self.gen:
            self.gen = gen
            self.x = self.x0 / gen ** self.deg
            # TODO: добавить возможность работы со списками или кортежами
        return self.x

class mixture:
    def __init__(self, methods, probs):
        self.methods = methods + [None]
        self.probs = probs + [1 - np.sum(probs)]
    def __call__(self, inds, args):
        m = np.random.choice(self.methods, p=self.probs)
        if m:
            m(inds, args)

class probOp:
    def __init__(self, method, prob):
        self.method = method
        self.prob = prob
    def __call__(self, inds, args):
        prob = evalf(self.prob, [args] + list(inds))
        if np.random.random() < self.prob:
            self.method(inds, args)

