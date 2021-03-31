import numpy as np
from random import random
from targets import randomRealVector
from alg_base import algorithm, evalf, fillAttribute, copyAttribute, simpleMove
from patterns import foreach, reducePop, evaluate
import copy

class particleSwarmOptimization(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opLimitVel = None
        self.alphabeta = None
        self.gamma = None
        self.delta = None
        algorithm.initAttributes(self, **args)

    @staticmethod
    def updateVel(ind, **xt):
        gamma = evalf(xt['gamma'], inds=[ind], **xt)
        alpha, beta = evalf(xt['alphabeta'], inds=[ind], **xt)
        g = xt['g']
        ind['v'] = gamma * ind['v'] + alpha * (ind['p'] - ind['x']) + beta * (g - ind['x'])

    @staticmethod
    def updateBestPosition(ind, **xt):
        goal = xt['goal']
        if goal.isBetter(ind['fNew'], ind['f']):
            ind['p'] = ind['x'].copy()
            ind['f'] = ind['fNew']

    def start(self):
        algorithm.start(self, "alphabeta gamma g", "x f fNew v p")
        foreach(self.population, self.opInit, key='x', **self.env)
        foreach(self.population, copyAttribute, keyFrom='x', keyTo='p', **self.env)
        evaluate(self.population, keyx='x', keyf='f', **self.env)
        vel = self.delta * (self.target.bounds[1] - self.target.bounds[0])
        foreach(self.population, fillAttribute(randomRealVector(dim=self.target.dimension, bounds=[-vel, vel])), key='v', **self.env)

    def __call__(self):
        self.start()
        while not self.stop(self.env):
            self.newGeneration()
            op = lambda x, y: x if self.goal.isBetter(x[1], y[1]) else y
            self.env['g'] = reducePop(self.population, lambda x: (x['p'], x['f']), op, lambda x: x[0], _t='reduce', **self.env)
            foreach(self.population, self.updateVel, _t='updatevel', **self.env)
            if self.opLimitVel != None:
                foreach(self.population, self.opLimitVel, key='v', _t='limitvel', **self.env)
            foreach(self.population, simpleMove, keyx='x', keyv='v', dt=1.0, _t='move', **self.env)
            evaluate(self.population, keyx='x', keyf='fNew', _t='evaluate', **self.env)
            foreach(self.population, self.updateBestPosition, _t='updatebest', **self.env)
        self.finish()

class randomAlphaBeta:
    def __init__(self, a, b=0):
        self.alpha = a
        self.beta = b if b > 0 else a
    def __call__(self, **xt):
        a = random() * self.alpha
        b = random() * self.beta
        return a, b

class linkedAlphaBeta:
    def __init__(self, t):
        self.total = t
    def __call__(self, **xt):
        a = random() * self.total
        b = self.total - a
        return a, b

class maxAmplitude:
    def __init__(self, amax):
        self.amax = amax
    def __call__(self, ind, **xt):
        key = xt['key']
        a = np.linalg.norm(ind[key])
        amax = evalf(self.amax, inds=[ind], **xt)
        if a > amax:
            ind[key] *= amax / a

class fixedAmplitude:
    def __init__(self, ampl):
        self.ampl = ampl
    def __call__(self, ind, **xt):
        key = xt['key']
        a = np.linalg.norm(ind[key])
        ampl = evalf(self.ampl, inds=[ind], **xt)
        ind[key] *= ampl / a
