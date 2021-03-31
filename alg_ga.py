from random import random, randrange
from alg_base import algorithm, evalf
from patterns import foreach, evaluate  
import copy

class geneticAlgorithm(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opSelect = lambda x, y: None
        self.opCrossover = lambda x, y: None
        self.opMutate = None
        algorithm.initAttributes(self, **args)

    def start(self):
        algorithm.start(self, "", "x f")
        foreach(self.population, self.opInit, key='x', **self.env) 
        evaluate(self.population, keyx='x', keyf='f', **self.env)

    def __call__(self):
        self.start()
        while not self.stop(self.env):
            self.newGeneration()
            self.opSelect(self.population, key='f', _t='select', **self.env) 
            self.opCrossover(self.population, key='x', _t='cross', **self.env)
            foreach(self.population, self.opMutate, key='x', _t='mutate', **self.env) 
            evaluate(self.population, keyx='x', keyf='f', _t='evaluate', **self.env)    
        self.finish()        

class tournament:
    def __init__(self, pwin):
        self.pwin = pwin
    def __call__(self, pair, **xt):
        ind1, ind2 = pair
        key = xt['key']
        twoway = xt['twoway']
        goal = xt['goal']
        pwin = evalf(self.pwin, inds=[ind1, ind2], **xt)
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
    def __call__(self, pair, **xt):
        ind1, ind2 = pair
        pswap = evalf(self.pswap, inds=[ind1, ind2], **xt)
        key = xt['key']
        dim = xt['target'].dimension
        for pos in range(dim):
            if random() < pswap:
                if xt['twoway']:
                    ind1[key][pos], ind2[key][pos] = ind2[key][pos], ind1[key][pos]
                else:
                    ind1[key][pos] = ind2[key][pos] 
                    
class singlePointCrossover:
    def __init__(self):
        pass
    def __call__(self, pair, **xt):
        ind1, ind2 = pair
        key = xt['key']
        dim = xt['target'].dimension
        cpos = randrange(1, dim)
        for pos in range(cpos, dim):
            if xt['twoway']:
                ind1[key][pos], ind2[key][pos] = ind2[key][pos], ind1[key][pos]
            else:
                ind1[key][pos] = ind2[key][pos]

class doublePointCrossover:
    def __init__(self):
        pass
    def __call__(self, pair, **xt):
        ind1, ind2 = pair
        key = xt['key']
        dim = xt['target'].dimension
        cpos1 = randrange(1, dim - 1)
        cpos2 = randrange(cpos1, dim)
        for pos in range(cpos1, cpos2):
            if xt['twoway']:
                ind1[key][pos], ind2[key][pos] = ind2[key][pos], ind1[key][pos]
            else:
                ind1[key][pos] = ind2[key][pos]

