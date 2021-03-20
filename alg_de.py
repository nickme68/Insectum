from alg_base import * 

class differentialEvolution(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opMakeProbe = None
        self.opCrossover = None
        self.opSelect = None
        self.weight = None
        algorithm.initAttributes(self, args)

    def start(self):
        algorithm.start(self, "weight", "x f", shadows="probes")
        foreach(self.population, self.opInit, self.args(key='x'))
        self.evaluateAll()

    def __call__(self):
        self.start()
        while not self.metrics.stopIt():
            self.newGeneration()
            pop2ind(self.probes, self.population, self.opMakeProbe, self.args(keyx='x'))
            pairwise(self.probes, self.population, self.opCrossover, self.args(key='x', twoway=False))
            evaluate(self.probes, self.args(keyx='x', keyf='f'))
            pairwise(self.population, self.probes, self.opSelect, self.args(key='f', twoway=False))

def argbestDE(population, args):
    keyf = args['keyf']
    if args['metrics'].task.getDir() == 'min':
        return min(enumerate(population), key=lambda x: x[1][keyf])[0]
    else:
        return max(enumerate(population), key=lambda x: x[1][keyf])[0]
    
def probesClassic(ind, population, args):
    keyx = args['keyx']
    index = args['index']
    weight = args['env']['weight']
    S = samplex(len(population), 3, [index])
    a, b, c = [population[i] for i in S]
    ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx])

def probesBest(ind, population, args):
    keyx = args['keyx']
    index = args['index']
    weight = args['env']['weight']
    i = argbestDE(population, args)
    S = [i] + samplex(len(population), 2, [index, i])
    a, b, c = [population[i] for i in S]
    ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx])

def probesCur2Best(ind, population, args):
    keyx = args['keyx']
    index = args['index']
    weight = args['env']['weight']
    i = argbestDE(population, args)
    S = [index, i] + samplex(len(population), 2, [index, i])
    cur, a, b, c = [population[i] for i in S]
    ind[keyx] = cur[keyx] + weight * (a[keyx] - cur[keyx] + b[keyx] - c[keyx])

def probesBest2(ind, population, args):
    keyx = args['keyx']
    index = args['index']
    weight = args['env']['weight']
    i = argbestDE(population, args)
    S = [i] + samplex(len(population), 4, [index, i])
    a, b, c, d, e = [population[i] for i in S]
    ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx] + d[keyx] - e[keyx])

def probesRandom5(ind, population, args):
    keyx = args['keyx']
    index = args['index']
    weight = args['env']['weight']
    S = samplex(len(population), 5, [index])
    a, b, c, d, e = [population[i] for i in S]
    ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx] + d[keyx] - e[keyx])

