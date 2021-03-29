import numpy as np
from random import random
from targets import randomRealVector
from alg_base import algorithm, evalf, fillAttribute, copyAttribute, simpleMove
from patterns import foreach, reducePop 
import copy

class particleSwarmOptimization(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opLimitVel = None
        self.alphabeta = None
        self.gamma = None
        self.delta = None
        algorithm.initAttributes(self, args)

    @staticmethod
    def updateVel(ind, args):
        gamma = evalf(args['env']['gamma'], [args, ind])
        alpha, beta = evalf(args['env']['alphabeta'], [args, ind])
        g = args['env']['g']
        ind['v'] = gamma * ind['v'] + alpha * (ind['p'] - ind['x']) + beta * (g - ind['x'])

    @staticmethod
    def updateBestPosition(ind, args):
        goal = args['env']['goal']
        if goal.isBetter(ind['fNew'], ind['f']):
            ind['p'] = ind['x'].copy()
            ind['f'] = ind['fNew']

    def start(self):
        algorithm.start(self, "alphabeta gamma g", "x f fNew v p")
        foreach(self.population, self.opInit, self.args(key='x'))
        foreach(self.population, copyAttribute, self.args(keyFrom='x', keyTo='p'))
        self.evaluateAll()
        vel = self.delta * (self.target.bounds[1] - self.target.bounds[0])
        foreach(self.population, fillAttribute(randomRealVector(dim=self.target.dimension, bounds=[-vel, vel])), self.args(key='v'))

    def __call__(self):
        self.start()
        while not self.stop(self.env):
            self.newGeneration()
            op = lambda x, y: x if self.goal.isBetter(x[1], y[1]) else y
            self.env['g'] = reducePop(self.population, lambda x: (x['p'], x['f']), op, lambda x: x[0])
            foreach(self.population, self.updateVel, self.args())
            if self.opLimitVel != None:
                foreach(self.population, self.opLimitVel, self.args(key='v'))
            foreach(self.population, simpleMove, self.args(x='x', v='v', dt=1.0))
            self.evaluateAll(keyf='fNew')
            foreach(self.population, self.updateBestPosition, self.args())

class randomAlphaBeta:
    def __init__(self, a, b=0):
        self.alpha = a
        self.beta = b if b > 0 else a
    def __call__(self, args):
        a = random() * self.alpha
        b = random() * self.beta
        return a, b

class linkedAlphaBeta:
    def __init__(self, t):
        self.total = t
    def __call__(self, args):
        a = random() * self.total
        b = self.total - a
        return a, b

class maxAmplitude:
    def __init__(self, amax):
        self.amax = amax
    def __call__(self, ind, args):
        key = args['key']
        a = np.linalg.norm(ind[key])
        amax = evalf(self.amax, [args, ind])
        if a > amax:
            ind[key] *= amax / a

class fixedAmplitude:
    def __init__(self, ampl):
        self.ampl = ampl
    def __call__(self, ind, args):
        key = args['key']
        a = np.linalg.norm(ind[key])
        ampl = evalf(self.ampl, [args, ind])
        ind[key] *= ampl / a
