from alg_base import * 

class particleSwapOptimization(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opLimitVel = None
        self.alphabeta = None
        self.gamma = None
        self.delta = None
        algorithm.initAttributes(self, args)

    @staticmethod
    def updateVel(ind, args):
        gamma = evalf(args['env']['gamma'], [args, ind])
        alpha, beta = evalf(args['env']['alphabeta'], [args, ind])
        g = args['metrics'].bestSolution
        ind['v'] = gamma * ind['v'] + alpha * (ind['p'] - ind['x']) + beta * (g - ind['x'])

    @staticmethod
    def updateBestPosition(ind, args):
        task = args['metrics'].task
        if task.isBetter(ind['fNew'], ind['f']):
            ind['p'] = ind['x'].copy()
            ind['f'] = ind['fNew']

    def start(self):
        algorithm.start(self, "alphabeta gamma", "x f fNew v p")
        foreach(self.population, self.opInit, self.args(key='x'))
        foreach(self.population, copyAttribute, self.args(keyFrom='x', keyTo='p'))
        self.evaluateAll()
        vel = self.delta * (self.metrics.task.bounds[1] - self.metrics.task.bounds[0])
        foreach(self.population, fillAttribute(randomRealVector(bounds=[-vel, vel])), self.args(key='v'))

    def __call__(self):
        self.start()
        while not self.metrics.stopIt():
            self.newGeneration()
            foreach(self.population, self.updateVel, self.args())
            if self.opLimitVel != None:
                foreach(self.population, self.opLimitVel, self.args(key='v'))
            foreach(self.population, simpleMove, self.args(x='x', v='v', dt=1.0))
            self.evaluateAll(keyf='fNew')
            foreach(self.population, self.updateBestPosition, self.args())

class maxAmplitude:
    def __init__(self, amax):
        self.amax = amax
    def __call__(self, ind, args):
        key = args['key']
        a = np.linalg.norm(ind[key])
        amax = evalf(self.amax, [args, ind])
        if a > amax:
            ind[key] *= amax / a

class fixedAmplitude:
    def __init__(self, ampl):
        self.ampl = ampl
    def __call__(self, ind, args):
        key = args['key']
        a = np.linalg.norm(ind[key])
        ampl = evalf(self.ampl, [args, ind])
        ind[key] *= ampl / a
