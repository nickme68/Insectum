from random import random
from numpy import exp
from alg_base import algorithm, evalf, copyAttribute
from patterns import foreach, evaluate  
import copy

class simulatedAnnealing(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opMove = None
        self.theta = None
        algorithm.initAttributes(self, **args)

    def start(self):
        algorithm.start(self, "theta", "&x xNew *f fNew")
        foreach(self.population, self.opInit, key='x', **self.env) 
        evaluate(self.population, keyx='x', keyf='f', **self.env)

    def runGeneration(self):
        foreach(self.population, copyAttribute, keyFrom='x', keyTo='xNew', _t='copy', **self.env) 
        foreach(self.population, self.opMove, key='xNew', _t='move', **self.env) 
        evaluate(self.population, keyx='xNew', keyf='fNew', _t='evaluate', **self.env)
        foreach(self.population, self.accept, _t='accept', **self.env) 

    @staticmethod
    def accept(ind, **xt):
        theta = evalf(xt['theta'], inds=[ind], **xt) 
        goal = xt['goal']
        df = abs(ind['fNew'] - ind['f'])
        if goal.isBetter(ind['fNew'], ind['f']) or random() < exp(-df / theta):
            ind['f'] = ind['fNew']
            ind['x'] = ind['xNew'].copy()

class expCool:
    def __init__(self, x0, q):
        self.x = x0
        self.q = q
        self.gen = 0
    def __call__(self, **xt):
        gen = xt['time']
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
    def __call__(self, **xt):
        gen = xt['time']
        if gen > self.gen:
            self.gen = gen
            self.x = self.x0 / gen ** self.deg
            # TODO: добавить возможность работы со списками или кортежами
        return self.x

class realMutation:
    def __init__(self, delta):
        self.delta = delta
    def __call__(self, ind, **xt):
        key = xt['key']
        delta = evalf(self.delta, inds=[ind], **xt) 
        dim = len(ind[key])
        for pos in range(dim):
            ind[key][pos] += delta * (1 - 2 * random())

class binaryMutation:
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, ind, **xt):
        key = xt['key']
        prob = evalf(self.prob, inds=[ind], **xt) 
        dim = len(ind[key])
        for pos in range(dim):
            if random() < prob:
                ind[key][pos] = 1 - ind[key][pos]