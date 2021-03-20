from alg_base import algorithm, evalf, samplex
from patterns import foreach, evaluate, pop2ind, pairs 

class differentialEvolution(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opMakeProbe = None
        self.opCrossover = None
        self.opSelect = None
        self.probes = None
        algorithm.initAttributes(self, args)

    def start(self):
        algorithm.start(self, "", "x f", shadows="probes")
        foreach(self.population, self.opInit, self.args(key='x'))
        self.evaluateAll()

    def __call__(self):
        self.start()
        while not self.stop(self.env):
            self.newGeneration()
            pop2ind(self.probes, self.population, self.opMakeProbe, 
                self.args(keyx='x', keyf='f'))
            pairs(self.probes, self.population, self.opCrossover, self.args(key='x'))
            evaluate(self.probes, self.args(keyx='x', keyf='f'))
            pairs(self.population, self.probes, self.opSelect, self.args(key='f'))

def argbestDE(population, args):
    keyf = args['keyf']
    if args['env']['goal'].getDir() == 'min':
        return min(enumerate(population), key=lambda x: x[1][keyf])[0]
    else:
        return max(enumerate(population), key=lambda x: x[1][keyf])[0]
    
class probesClassic:
    def __init__(self, weight):
        self.weight = weight
    def __call__(self, ind, population, args):
        keyx = args['keyx']
        index = args['index']
        weight = evalf(self.weight, [args, ind])
        S = samplex(len(population), 3, [index])
        a, b, c = [population[i] for i in S]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx])

class probesBest:
    def __init__(self, weight):
        self.weight = weight
    def __call__(self, ind, population, args):
        keyx = args['keyx']
        index = args['index']
        weight = evalf(self.weight, [args, ind])
        i = argbestDE(population, args)
        S = [i] + samplex(len(population), 2, [index, i])
        a, b, c = [population[i] for i in S]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx])

class probesCur2Best:
    def __init__(self, weight):
        self.weight = weight
    def __call__(self, ind, population, args):
        keyx = args['keyx']
        index = args['index']
        weight = evalf(self.weight, [args, ind])
        i = argbestDE(population, args)
        S = [index, i] + samplex(len(population), 2, [index, i])
        cur, a, b, c = [population[i] for i in S]
        ind[keyx] = cur[keyx] + weight * (a[keyx] - cur[keyx] + b[keyx] - c[keyx])

class probesBest2:
    def __init__(self, weight):
        self.weight = weight
    def __call__(self, ind, population, args):
        keyx = args['keyx']
        index = args['index']
        weight = evalf(self.weight, [args, ind])
        i = argbestDE(population, args)
        S = [i] + samplex(len(population), 4, [index, i])
        a, b, c, d, e = [population[i] for i in S]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx] + d[keyx] - e[keyx])

class probesRandom5:
    def __init__(self, weight):
        self.weight = weight
    def __call__(self, ind, population, args):
        keyx = args['keyx']
        index = args['index']
        weight = evalf(self.weight, [args, ind])
        S = samplex(len(population), 5, [index])
        a, b, c, d, e = [population[i] for i in S]
        ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx] + d[keyx] - e[keyx])

