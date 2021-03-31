import numpy as np
from targets import randomRealVector
from alg_base import algorithm, evalf, fillAttribute, shuffled
from patterns import evaluate, foreach, reducePop 

class competitiveSwarmOptimizer(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.socialFactor = None
        self.delta = None
        algorithm.initAttributes(self, **args)

    @staticmethod
    def tournament(pair, **xt):
        ind1, ind2 = pair
        dim = xt['target'].dimension
        x = xt['x']
        phi = evalf(xt['socialFactor'], inds=[ind1, ind2], **xt)

        if xt['goal'].isBetter(ind1['f'], ind2['f']):
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
        foreach(self.population, self.opInit, key='x', **self.env)
        evaluate(self.population, keyx='x', keyf='f', **self.env)
        vel = self.delta * (self.target.bounds[1] - self.target.bounds[0])
        foreach(self.population, fillAttribute(randomRealVector(dim=self.target.dimension, bounds=[-vel, vel])), key='v', **self.env)
        self.compete = shuffled(self.tournament)

    def __call__(self):
        self.start()
        while not self.stop(self.env):
            self.newGeneration()
            self.env['x'] = reducePop(self.population, lambda x: x['x'], np.add, lambda x: x / self.popSize, _t='reduce', **self.env)
            self.compete(self.population, _t='compete', **self.env) 
            evaluate(self.population, keyx='x', keyf='f', reEval='reEval', _t='evaluate', **self.env)
        self.finish()
