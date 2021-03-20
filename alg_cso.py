import numpy as np
from targets import randomRealVector
from alg_base import algorithm, evalf, fillAttribute, shuffled
from patterns import evaluate, foreach, reducePop 

class competitiveSwarmOptimizer(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.socialFactor = None
        self.delta = None
        algorithm.initAttributes(self, args)

    @staticmethod
    def tournament(pair, args):
        ind1, ind2 = pair
        dim = args['env']['target'].dimension
        x = args['env']['x']
        phi = evalf(args['env']['socialFactor'], [args, ind1, ind2])

        firstWin = args['env']['goal'].isBetter(ind1['f'], ind2['f'])
        ind1['reEval'] = not firstWin
        ind2['reEval'] = firstWin
        R = np.random.rand(3, dim)

        if firstWin:
            ind2['v'] = np.multiply(R[0], ind2['v']) + np.multiply(R[1], ind1['x'] - ind2['x']) + np.multiply(R[2], phi * (x - ind2['x']))
            ind2['x'] += ind2['v']
        else:
            ind1['v'] = np.multiply(R[0], ind1['v']) + np.multiply(R[1], ind2['x'] - ind1['x']) + np.multiply(R[2], phi * (x - ind1['x']))
            ind1['x'] += ind1['v']

    def start(self):
        algorithm.start(self, "socialFactor x", "x f v reEval")
        foreach(self.population, self.opInit, self.args(key='x'))
        self.evaluateAll()
        vel = self.delta * (self.target.bounds[1] - self.target.bounds[0])
        foreach(self.population, fillAttribute(randomRealVector(dim=self.target.dimension, bounds=[-vel, vel])), self.args(key='v'))
        self.compete = shuffled(self.tournament)

    def __call__(self):
        self.start()
        while not self.stop(self.env):
            self.newGeneration()
            self.env['x'] = reducePop(self.population, lambda x: x['x'], np.add, lambda x: x / self.popSize)
            self.compete(self.population, self.args()) 
            evaluate(self.population, self.args(keyx='x', keyf='f', reEval='reEval'))
