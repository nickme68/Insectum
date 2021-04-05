from alg_base import algorithm, evalf, samplex
from patterns import foreach, evaluate, pop2ind, pairs 

class differentialEvolution(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opMakeProbe = None
        self.opCrossover = None
        self.opSelect = None
        self.probes = None
        algorithm.initAttributes(self, **args)

    def start(self):
        algorithm.start(self, "", "x f", shadows="probes")
        foreach(self.population, self.opInit, key='x', **self.env)
        evaluate(self.population, keyx='x', keyf='f', **self.env)

    def __call__(self):
        self.start()
        while not self.stop(self.env):
            self.newGeneration()
            pop2ind(self.probes, self.population, self.opMakeProbe, 
                keyx='x', keyf='f', _t='makeprobes', **self.env)
            pairs(self.probes, self.population, self.opCrossover, key='x', _t='crossover', **self.env)
            evaluate(self.probes, keyx='x', keyf='f', _t='evaluate', **self.env)
            pairs(self.population, self.probes, self.opSelect, key='f', _t='select', **self.env)
        self.finish()

def argbestDE(population, **xt):
    keyf = xt['keyf']
    if xt['goal'].getDir() == 'min':
        return min(enumerate(population), key=lambda x: x[1][keyf])[0]
    else:
        return max(enumerate(population), key=lambda x: x[1][keyf])[0]
    
class probeClassic:
    def __init__(self, weight):
        self.weight = weight
    def __call__(self, ind, population, **xt):
        keyx = xt['keyx']
        index = xt['index']
        weight = evalf(self.weight, inds=[ind], **xt)
        S = samplex(len(population), 3, [index])
        a, b, c = [population[i] for i in S]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx])

class probeBest:
    def __init__(self, weight):
        self.weight = weight
    def __call__(self, ind, population, xt):
        keyx = xt['keyx']
        index = xt['index']
        weight = evalf(self.weight, inds=[ind], **xt)
        i = argbestDE(population, **xt)
        S = [i] + samplex(len(population), 2, [index, i])
        a, b, c = [population[i] for i in S]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx])

class probeCur2Best:
    def __init__(self, weight):
        self.weight = weight
    def __call__(self, ind, population, **xt):
        keyx = xt['keyx']
        index = xt['index']
        weight = evalf(self.weight, inds=[ind], **xt)
        i = argbestDE(population, **xt)
        S = [index, i] + samplex(len(population), 2, [index, i])
        cur, a, b, c = [population[i] for i in S]
        ind[keyx] = cur[keyx] + weight * (a[keyx] - cur[keyx] + b[keyx] - c[keyx])

class probeBest2:
    def __init__(self, weight):
        self.weight = weight
    def __call__(self, ind, population, **xt):
        keyx = xt['keyx']
        index = xt['index']
        weight = evalf(self.weight, inds=[ind], **xt)
        i = argbestDE(population, **xt)
        S = [i] + samplex(len(population), 4, [index, i])
        a, b, c, d, e = [population[i] for i in S]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx] + d[keyx] - e[keyx])

class probeRandom5:
    def __init__(self, weight):
        self.weight = weight
    def __call__(self, ind, population, **xt):
        keyx = xt['keyx']
        index = xt['index']
        weight = evalf(self.weight, inds=[ind], **xt)
        S = samplex(len(population), 5, [index])
        a, b, c, d, e = [population[i] for i in S]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx] + d[keyx] - e[keyx])

