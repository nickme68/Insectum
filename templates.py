import numpy as np 

def evaluate(population, args):
    metrics = args['metrics']
    keyx = args['keyx']
    keyf = args['keyf']
    for ind in population:
        metrics.newEval(ind, keyx, keyf)

def foreach(population, op, args):
    for ind in population:
        op(ind, args)

class shuffled:
    def __init__(self, op):
        self.op = op
    def __call__(self, population, args):
        popSize = len(population)
        P = list(range(popSize))
        np.random.shuffle(P)
        for i in range(popSize // 2):
            ind1, ind2 = population[P[2 * i]], population[P[2 * i + 1]]
            self.op((ind1, ind2), args)

