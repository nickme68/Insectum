from random import random
from numpy import exp
from alg_base import algorithm, evalf, copyAttribute
from patterns import foreach  
import copy

class simulatedAnnealing(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opMove = None
        self.theta = None
        algorithm.initAttributes(self, args)

    def start(self):
        algorithm.start(self, "theta", "x xNew f fNew")
        foreach(self.population, self.opInit, self.args(key='x'))
        self.evaluateAll()

    def __call__(self):
        self.start()
        while not self.stop(self.env):
            self.newGeneration()
            foreach(self.population, copyAttribute, self.args(keyFrom='x', keyTo='xNew'))
            foreach(self.population, self.opMove, self.args(key='xNew'))
            self.evaluateAll(keyx='xNew', keyf='fNew')
            foreach(self.population, self.accept, self.args())
            
    @staticmethod
    def accept(ind, args):
        theta = evalf(args['env']['theta'], [args, ind])
        goal = args['env']['goal']
        df = abs(ind['fNew'] - ind['f'])
        if goal.isBetter(ind['fNew'], ind['f']) or random() < exp(-df / theta):
            ind['f'] = ind['fNew']
            ind['x'] = ind['xNew'].copy()

class expCool:
    def __init__(self, x0, q):
        self.x = x0
        self.q = q
        self.gen = 0
    def __call__(self, args):
        gen = args[0]['env']['time']
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
        gen = args[0]['env']['time']
        if gen == 0:
            return self.x
        if gen > self.gen:
            self.gen = gen
            self.x = self.x0 / gen ** self.deg
            # TODO: добавить возможность работы со списками или кортежами
        return self.x

class realMutation:
    def __init__(self, delta):
        self.delta = delta
    def __call__(self, ind, args):
        key = args['key']
        delta = evalf(self.delta, [args, ind])
        dim = len(ind[key])
        for pos in range(dim):
            ind[key][pos] += delta * (1 - 2 * random())

class binaryMutation:
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, ind, args):
        key = args['key']
        prob = evalf(self.prob, [args, ind])
        dim = len(ind[key])
        for pos in range(dim):
            if random() < prob:
                ind[key][pos] = 1 - ind[key][pos]