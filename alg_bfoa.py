from alg_base import * 

class bacterialForagingAlgorithm(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opSelect = None
        self.opDisperse = None
        self.opSignal = None
        self.vel = None
        self.gamma = None
        self.probRotate = None
        self.mu = None
        algorithm.initAttributes(self, args)

    @staticmethod
    def initVel(ind, args):
        dim = args['metrics'].task.dimension
        vel = evalf(args['env']['vel'], [args, ind])
        ind['v'] = randomDirectedVector(dim, vel)

    @staticmethod
    def rotate(ind, args):
        prob = evalf(args['env']['probRotate'], [args, ind])
        newBetter = args['metrics'].task.isBetter(ind['fNew'], ind['f'])
        r = np.random.random()
        if newBetter and r < prob[0] or not newBetter and r < prob[1]:
            vel = evalf(args['env']['vel'], [args, ind])
            dim = args['metrics'].task.dimension
            ind['v'] = randomDirectedVector(dim, vel)

    @staticmethod
    def updateFs(ind, args):
        gamma = evalf(args['env']['gamma'], [args, ind])
        ind['f'] = ind['fNew']
        ind['fTotal'] = (gamma * ind['fTotal'] + ind['fNew']) / (gamma + 1)

    def start(self):
        algorithm.start(self, "vel gamma probRotate", "x f fNew fs fTotal v")
        foreach(self.population, self.opInit, self.args(key='x'))
        self.evaluateAll()
        foreach(self.population, self.initVel, self.args())
        foreach(self.population, copyAttribute, self.args(keyFrom='f', keyTo='fTotal'))
        foreach(self.population, fillAttribute(0.0), self.args(key='fs'))

    def __call__(self):
        self.start()
        while not self.metrics.stopIt():
            self.newGeneration()
            foreach(self.population, simpleMove, self.args(x='x', v='v', dt=1.0))
            self.evaluateAll(keyf='fNew')
            # signaling!
            self.opSignal(self.population, self.args(keyx='x', keys='fs'))
            foreach(self.population, simpleMove, self.args(x='fNew', v='fs', dt=self.mu))
            foreach(self.population, self.rotate, self.args())
            foreach(self.population, self.updateFs, self.args())
            self.opSelect(self.population, self.args(key='fTotal'))
            foreach(self.population, self.opDisperse, self.args(key='x'))

def randomDirectedVector(dim, length):
    vec = np.random.normal(0.0, 1.0, size=dim)
    return vec * (length / np.linalg.norm(vec))

class calcSignals:
    def __init__(self, dt, signalShape, signalsReduce):
        self.dt = dt
        self.signalShape = signalShape
        self.signalsReduce = signalsReduce
    def __call__(self, population, args):
        if args['metrics'].currentGeneration % self.dt > 0:
            return
        keyx = args['keyx']
        keys = args['keys']
        n = len(population)
        D = np.zeros((n, n))
        for i in range(n - 1):
            for j in range(i+1, n):
                D[i, j] = D[j, i] = np.linalg.norm(population[i][keyx] - population[j][keyx])
        for i in range(n):
            ind = population[i]
            signals = self.signalShape(D[i])
            if self.signalsReduce == "sum":
                ind[keys] = np.sum(signals)
            elif self.signalsReduce == "min":
                ind[keys] = np.min(signals)
            else:
                ind[keys] = self.signalsReduce(signals)

# Different signal shapes

class signalClustering:
    def __init__(self, d, direction):
        self.d = d
        self.direction = -1 if direction == "min" else 1
    def __call__(self, x):
        x2 = (x / self.d) ** 2
        return self.direction * (2 * np.exp(-x2) - 3 * np.exp(-4 * x2))