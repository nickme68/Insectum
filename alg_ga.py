from random import random, randrange
from alg_base import algorithm, evalf
from patterns import foreach  
import copy

class geneticAlgorithm(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opSelect = lambda x, y: None
        self.opCrossover = lambda x, y: None
        self.opMutate = None
        algorithm.initAttributes(self, args)

    def start(self):
        algorithm.start(self, "", "x f")
        foreach(self.population, self.opInit, self.args(key='x'))
        self.evaluateAll()

    def __call__(self):
        self.start()
        while not self.stop(self.env):
            self.newGeneration()
            self.opSelect(self.population, self.args(key='f'))
            self.opCrossover(self.population, self.args(key='x'))
            foreach(self.population, self.opMutate, self.args(key='x'))
            self.evaluateAll()

class tournament:
    def __init__(self, pwin):
        self.pwin = pwin
    def __call__(self, pair, args):
        ind1, ind2 = pair
        key = args['key']
        twoway = args['twoway']
        goal = args['env']['goal']
        pwin = evalf(self.pwin, [args, ind1, ind2])
        A = goal.isBetter(ind1[key], ind2[key])
        B = random() < pwin
        if A != B: # xor
            ind1.update(copy.deepcopy(ind2))
        elif twoway:
            ind2.update(copy.deepcopy(ind1))

# Crossovers

class uniformCrossover:
    def __init__(self, pswap):
        self.pswap = pswap
    def __call__(self, pair, args):
        ind1, ind2 = pair
        pswap = evalf(self.pswap, [args, ind1, ind2])
        key = args['key']
        dim = args['env']['target'].dimension
        for pos in range(dim):
            if random() < pswap:
                if args['twoway']:
                    ind1[key][pos], ind2[key][pos] = ind2[key][pos], ind1[key][pos]
                else:
                    ind1[key][pos] = ind2[key][pos] 
                    
class singlePointCrossover:
    def __init__(self):
        pass
    def __call__(self, pair, args):
        ind1, ind2 = pair
        key = args['key']
        dim = args['env']['target'].dimension
        cpos = randrange(1, dim)
        for pos in range(cpos, dim):
            if args['twoway']:
                ind1[key][pos], ind2[key][pos] = ind2[key][pos], ind1[key][pos]
            else:
                ind1[key][pos] = ind2[key][pos]

class doublePointCrossover:
    def __init__(self):
        pass
    def __call__(self, pair, args):
        ind1, ind2 = pair
        key = args['key']
        dim = args['env']['target'].dimension
        cpos1 = randrange(1, dim - 1)
        cpos2 = randrange(cpos1, dim)
        for pos in range(cpos1, cpos2):
            if args['twoway']:
                ind1[key][pos], ind2[key][pos] = ind2[key][pos], ind1[key][pos]
            else:
                ind1[key][pos] = ind2[key][pos]

