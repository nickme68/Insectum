from alg_base import * 

class differentialEvolution(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opMakeProbes = None
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
            pop2ind(self.probes, self.population, self.opMakeProbes, self.args(keyx='x', keyf='f'))
            pairwise(self.probes, self.population, self.opCrossover, self.args(key='x', twoway=False))
            evaluate(self.probes, self.args(keyx='x', keyf='f'))
            pairwise(self.population, self.probes, self.opSelect, self.args(key='f', twoway=False))

def sampleDE(n, m, x):
    s = []
    for i in range(n):
        if i not in x:
            s.append(i)
    return list(np.random.choice(s, m, False))

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
    S = sampleDE(len(population), 3, [index])
    a, b, c = [population[i] for i in S]
    ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx])

def probesBest(ind, population, args):
    keyx = args['keyx']
    index = args['index']
    weight = args['env']['weight']
    i = argbestDE(population, args)
    S = [i] + sampleDE(len(population), 2, [index, i])
    a, b, c = [population[i] for i in S]
    ind[keyx] = a[keyx] + weight * (b[keyx] - c[keyx])
