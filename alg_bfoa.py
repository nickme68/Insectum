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
        algorithm.initAttributes(self, args)

    @staticmethod
    def initVel(ind, args):
        dim = args['env']['target'].dimension
        vel = evalf(args['env']['vel'], [args, ind])
        ind['v'] = randomDirectedVector(dim, vel)

    @staticmethod
    def rotate(ind, args):
        prob = evalf(args['env']['probRotate'], [args, ind])
        newBetter = args['env']['goal'].isBetter(ind['fNew'], ind['f'])
        r = random()
        if newBetter and r < prob[0] or not newBetter and r < prob[1]:
            vel = evalf(args['env']['vel'], [args, ind])
            dim = args['env']['target'].dimension
            ind['v'] = randomDirectedVector(dim, vel)

    @staticmethod
    def updateF(ind, args):
        gamma = evalf(args['env']['gamma'], [args, ind])
        ind['f'] = ind['fNew']
        ind['fTotal'] = (gamma * ind['fTotal'] + ind['fNew']) / (gamma + 1)

    def start(self):
        algorithm.start(self, "vel gamma probRotate", "x f fNew fs fTotal v")
        foreach(self.population, self.opInit, self.args(key='x'))
        self.evaluateAll()
        foreach(self.population, self.initVel, self.args())
        foreach(self.population, copyAttribute, self.args(keyFrom='f', keyTo='fTotal'))
        foreach(self.population, fillAttribute(0.0), self.args(key='fs'))

    def __call__(self):
        self.start()
        while not self.stop(self.env):
            self.newGeneration()
            foreach(self.population, simpleMove, self.args(x='x', v='v', dt=1.0))
            self.evaluateAll(keyf='fNew')
            self.opSignals(self.population, self.args(keyx='x', keys='fs'))
            foreach(self.population, simpleMove, self.args(x='fNew', v='fs', dt=self.mu))
            foreach(self.population, self.rotate, self.args())
            foreach(self.population, self.updateF, self.args())
            self.opSelect(self.population, self.args(key='fTotal'))
            foreach(self.population, self.opDisperse, self.args(key='x'))

def randomDirectedVector(dim, length):
    vec = np.random.normal(0.0, 1.0, size=dim)
    return vec * (length / np.linalg.norm(vec))


def l2metrics(x, y):
    return np.linalg.norm(x - y)

class noSignals:
    def __call__(self, population, args):
        return None

class calcSignals:
    def __init__(self, shape, reduce=np.sum, metrics=l2metrics):
        self.shape = shape
        R = {"sum": np.sum, "min": np.min, "max": np.max, "mean": np.mean}
        self.reduce = reduce
        if reduce in R:
            self.reduce = R[reduce]
        self.metrics = metrics
    def __call__(self, population, args):
        keyx = args['keyx']
        keys = args['keys']
        n = len(population)
        D = np.zeros((n, n))
        for i in range(n - 1):
            for j in range(i+1, n):
                D[i, j] = D[j, i] = self.metrics(population[i][keyx], population[j][keyx])
        for i in range(n):
            ind = population[i]
            S = np.zeros(n)
            for j in range(n):
                S[j] = self.shape(D[i][j], args, population[i], population[j]) 
            ind[keys] = self.reduce(S)

# Different signal shapes

class signalClustering:
    def __init__(self, d, direction):
        self.d = d
        self.direction = -1 if direction == "min" else 1
    def __call__(self, x, *args):
        d = evalf(self.d, args)
        x2 = (x / d) ** 2
        return self.direction * (2 * np.exp(-x2) - 3 * np.exp(-4 * x2))