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

        if args['env']['goal'].isBetter(ind1['f'], ind2['f']):
            winner, loser = ind1, ind2
        else:
            winner, loser = ind2, ind1
        winner['reEval'], loser['reEval'] = False, True
        R = np.random.rand(3, dim)
        loser['v'] = np.multiply(R[0], loser['v']) 
        loser['v'] += np.multiply(R[1], winner['x'] - loser['x'])
        loser['v'] += np.multiply(R[2], phi * (x - loser['x']))
        loser['x'] += loser['v']

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
