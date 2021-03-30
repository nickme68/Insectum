import numpy as np
from random import random
from targets import randomRealVector
from alg_base import algorithm, evalf, fillAttribute, simpleMove, copyAttribute
from patterns import foreach 

class bacterialForagingAlgorithm(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opSelect = lambda x, y: None
        self.opDisperse = None
        self.opSignals = lambda x, y: None
        self.vel = None
        self.gamma = None
        self.probRotate = None
        self.mu = None
        algorithm.initAttributes(self, **args)

    @staticmethod
    def initVel(ind, **xt):
        dim = xt['target'].dimension
        vel = evalf(xt['vel'], inds=[ind], **xt)
        ind['v'] = randomDirectedVector(dim, vel)

    @staticmethod
    def rotate(ind, **xt):
        prob = evalf(xt['probRotate'], inds=[ind], **xt)
        newBetter = xt['goal'].isBetter(ind['fNew'], ind['f'])
        r = random()
        if newBetter and r < prob[0] or not newBetter and r < prob[1]:
            vel = evalf(xt['vel'], inds=[ind], **xt)
            dim = xt['target'].dimension
            ind['v'] = randomDirectedVector(dim, vel)

    @staticmethod
    def updateF(ind, **xt):
        gamma = evalf(xt['gamma'], inds=[ind], **xt)
        ind['f'] = ind['fNew']
        ind['fTotal'] = (gamma * ind['fTotal'] + ind['fNew']) / (gamma + 1)

    def start(self):
        algorithm.start(self, "vel gamma probRotate", "x f fNew fs fTotal v")
        foreach(self.population, self.opInit, key='x', **self.env)
        self.evaluateAll()
        foreach(self.population, self.initVel, **self.env)
        foreach(self.population, copyAttribute, keyFrom='f', keyTo='fTotal', **self.env)
        foreach(self.population, fillAttribute(0.0), key='fs', **self.env)

    def __call__(self):
        self.start()
        while not self.stop(self.env):
            self.newGeneration()
            foreach(self.population, simpleMove, keyx='x', keyv='v', dt=1.0, **self.env)
            self.evaluateAll(keyf='fNew')
            self.opSignals(self.population, keyx='x', keys='fs', **self.env)
            foreach(self.population, simpleMove, keyx='fNew', keyv='fs', dt=self.mu, **self.env)
            foreach(self.population, self.rotate, **self.env)
            foreach(self.population, self.updateF, **self.env)
            self.opSelect(self.population, key='fTotal', **self.env)
            foreach(self.population, self.opDisperse, key='x', **self.env)

def randomDirectedVector(dim, length):
    vec = np.random.normal(0.0, 1.0, size=dim)
    return vec * (length / np.linalg.norm(vec))
    
def l2metrics(x, y):
    return np.linalg.norm(x - y)

class noSignals:
    def __call__(self, population, **xt):
        return None

class calcSignals:
    def __init__(self, shape, reduce=np.sum, metrics=l2metrics):
        self.shape = shape
        R = {"sum": np.sum, "min": np.min, "max": np.max, "mean": np.mean}
        self.reduce = reduce
        if reduce in R:
            self.reduce = R[reduce]
        self.metrics = metrics
    def __call__(self, population, **xt):
        keyx = xt['keyx']
        keys = xt['keys']
        n = len(population)
        D = np.zeros((n, n))
        for i in range(n - 1):
            for j in range(i+1, n):
                D[i, j] = D[j, i] = self.metrics(population[i][keyx], population[j][keyx])
        for i in range(n):
            ind = population[i]
            S = np.zeros(n)
            for j in range(n):
                S[j] = self.shape(D[i][j], inds=[population[i], population[j]], **xt) 
            ind[keys] = self.reduce(S)

# Different signal shapes

class signalClustering:
    def __init__(self, d, direction):
        self.d = d
        self.direction = -1 if direction == "min" else 1
    def __call__(self, x, **xt):
        d = evalf(self.d, **xt)
        x2 = (x / d) ** 2
        return self.direction * (2 * np.exp(-x2) - 3 * np.exp(-4 * x2))