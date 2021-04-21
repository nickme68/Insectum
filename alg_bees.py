import numpy as np
from random import random
from alg_base import algorithm, rankIt
from patterns import foreach, evaluate, pop2ind

class beesAlgorithm(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.beesNum = None
        self.opLocal = None
        self.opGlobal = None
        self.opProbs = None
        self.opFly = None
        algorithm.initAttributes(self, **args)
        rankIt()(self)

    def start(self):
        algorithm.start(self, "", "&x *f")
        bee = {'x':None, 'f':None, '_rank':None}
        self.bees = [bee.copy() for i in range(self.beesNum)]
        self.opFly = opFly(self.opProbs, self.opLocal, self.opGlobal, self.popSize)
        foreach(self.population, self.opInit, key='x', **self.env)
        evaluate(self.population, keyx='x', keyf='f', **self.env)

    @staticmethod
    def updatePlace(place, bees, **xt):
        for bee in bees:
            if bee['_rank'] == place['_rank'] and xt['goal'].isBetter(bee['f'], place['f']):
                place['f'] = bee['f']
                place['x'] = bee['x'].copy()

    def runGeneration(self):
        pop2ind(self.bees, self.population, self.opFly, key='x', _t='fly', **self.env)
        evaluate(self.bees, keyx='x', keyf='f', _t='evaluate', **self.env)
        pop2ind(self.population, self.bees, self.updatePlace, _t='update', **self.env)

class opFly:
    def __init__(self, opProbs, opLocal, opGlobal, psize):
        self.opLocal = opLocal
        self.opGlobal = opGlobal
        self.probs = opProbs(psize)
    def __call__(self, bee, places, **xt):
        r = random()
        for pl in places:
            pr = self.probs[pl['_rank']]
            if r < pr:
                bee['_rank'] = pl['_rank']
                if pl['_rank'] == len(places) - 1:
                    self.opGlobal(bee, **xt)
                else:
                    bee['x'] = pl['x'].copy()
                    self.opLocal(bee, **xt)
                return
            r -= pr

class uniformPlacesProbs:
    def __init__(self, pscout):
        self.pscout = pscout 
    def __call__(self, size):
        probs = np.full(size, (1 - self.pscout) / (size - 1) )
        probs[size - 1] = self.pscout
        return probs

class linearPlacesProbs:
    def __init__(self, elitism, pscout):
        self.elitism = elitism
        self.pscout = pscout
    def __call__(self, size):
        a = self.elitism * (1 - self.pscout) * 2 / ((size - 1) * (size - 2))
        b = (1 - self.pscout) / (size - 1) + a * (size - 2) / 2
        probs = -a * np.array(range(size)) + b
        probs[size - 1] = self.pscout
        return probs

class binaryPlacesProbs:
    def __init__(self, rho, elitism, pscout):
        self.rho = rho
        self.mu = 1 / (1 - elitism)
        self.pscout = pscout
    def __call__(self, size):
        me = int((size - 1) * self.rho)
        mo = size - 1 - me
        pe = (1 - self.pscout) / (me + mo / self.mu)
        po = pe / self.mu
        return np.array([pe] * me + [po] * mo + [self.pscout])