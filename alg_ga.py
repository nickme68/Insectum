from alg_base import * 

class geneticAlgorithm(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opSelect = None
        self.opCrossover = None
        self.opMutate = None
        algorithm.initAttributes(self, args)

    def start(self):
        algorithm.start(self, "", "x f")
        foreach(self.population, self.opInit, self.args(key='x'))
        self.evaluateAll()

    def __call__(self):
        self.start()
        while not self.metrics.stopIt():
            self.newGeneration()
            self.opSelect(self.population, self.args(key='f'))
            self.opCrossover(self.population, self.args(key='x'))
            foreach(self.population, self.opMutate, self.args(key='x'))
            self.evaluateAll()

class tournament:
    def __init__(self, pwin, twoway=True):
        self.pwin = pwin
        self.twoway = twoway
    def __call__(self, pair, args):
        ind1, ind2 = pair
        key = args['key']
        twoway = 'twoway' not in args and self.twoway or 'twoway' in args and args['twoway']
        task = args['metrics'].task
        pwin = evalf(self.pwin, [args, ind1, ind2])
        A = task.isBetter(ind1[key], ind2[key])
        B = np.random.random() < pwin
        if A != B: # xor
            ind1.update(copy.deepcopy(ind2))
        elif twoway:
            ind2.update(copy.deepcopy(ind1))

# Crossovers

class uniformCrossover:
    def __init__(self, pswap, twoway=True):
        self.pswap = pswap
        self.twoway = twoway
    def __call__(self, pair, args):
        ind1, ind2 = pair
        pswap = evalf(self.pswap, [args, ind1, ind2])
        key = args['key']
        twoway = 'twoway' not in args and self.twoway or 'twoway' in args and args['twoway']
        dim = args['metrics'].task.dimension
        for pos in range(dim):
            if np.random.random() < pswap:
                if twoway:
                    ind1[key][pos], ind2[key][pos] = ind2[key][pos], ind1[key][pos]
                else:
                    ind1[key][pos] = ind2[key][pos] 
class singlePointCrossover:
    def __init__(self, twoway=True):
        self.twoway = twoway
    def __call__(self, pair, args):
        ind1, ind2 = pair
        key = args['key']
        twoway = 'twoway' not in args and self.twoway or 'twoway' in args and args['twoway']
        dim = args['metrics'].task.dimension
        cpos = np.random.randint(low=1, high=dim)
        for pos in range(cpos, dim):
            if twoway:
                ind1[key][pos], ind2[key][pos] = ind2[key][pos], ind1[key][pos]
            else:
                ind1[key][pos] = ind2[key][pos]

class doublePointCrossover:
    def __init__(self, twoway=True):
        self.twoway = twoway
    def __call__(self, pair, args):
        ind1, ind2 = pair
        key = args['key']
        twoway = 'twoway' not in args and self.twoway or 'twoway' in args and args['twoway']
        dim = args['metrics'].task.dimension
        cpos1 = np.random.randint(low=1, high=dim - 1)
        cpos2 = np.random.randint(low=cpos1, high=dim)
        for pos in range(cpos1, cpos2):
            if twoway:
                ind1[key][pos], ind2[key][pos] = ind2[key][pos], ind1[key][pos]
            else:
                ind1[key][pos] = ind2[key][pos]

# Mutations (unary operator)

class realMutation:
    def __init__(self, delta):
        self.delta = delta
    def __call__(self, ind, args):
        key = args['key']
        dim = args['metrics'].task.dimension
        delta = evalf(self.delta, [args, ind])
        for pos in range(dim):
            ind[key][pos] += np.random.uniform(low=-delta, high=delta)

class binaryMutation:
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, ind, args):
        key = args['key']
        dim = args['metrics'].task.dimension
        prob = evalf(self.prob, [args, ind])
        for pos in range(dim):
            if np.random.random() < prob:
                ind[key][pos] = 1 - ind[key][pos]