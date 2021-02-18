from alg_base import * 

class beesAlgorithm(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opFlight = None
        self.opPlaceProbs = None
        self.plNum = None
        self.probScout = None
        algorithm.initAttributes(self, args)

    def sortPlaces(self):
        if self.metrics.task.getDir() == "min":
            self.env['places'].sort(key=lambda x: x['f'])
        else:
            self.env['places'].sort(key=lambda x: -x['f'])

    @staticmethod
    def updatePlace(ind, args):
        isBetter = args['metrics'].task.isBetter
        pl = args['env']['places'][ind['p']]
        if isBetter(ind['f'], pl['f']):
            pl['f'] = ind['f']
            pl['x'] = ind['x'].copy()

    def start(self):
        algorithm.start(self, "places probs plNum", "x f p")
        pl = {'x':None, 'f':None}
        self.env['places'] = [pl.copy() for i in range(self.plNum + 1)]
        foreach(self.env['places'], self.opInit, self.args(key='x'))
        evaluate(self.env['places'], self.args(keyx='x', keyf='f'))
        self.sortPlaces()
        self.env['probs'] = self.opPlaceProbs(self.plNum, self.probScout)

    def __call__(self):
        self.start()
        while not self.metrics.stopIt():
            self.newGeneration()
            foreach(self.population, self.opFlight, self.args(key='x'))
            self.evaluateAll()
            foreach(self.population, self.updatePlace, self.args()) # TODO: replace by reduce!
            self.sortPlaces()

class beeFlight: 
    def __init__(self, opLocal, opGlobal):
        self.loc = opLocal
        self.glob = opGlobal
    def __call__(self, ind, args):
        env = args['env']
        m = env['plNum']
        probs = env['probs']
        p = np.random.choice(range(m + 1), p=probs)
        ind['p'] = p
        if p < m:
            ind['x'] = env['places'][p]['x'].copy()
            self.loc(ind, args)
        else:
            self.glob(ind, args)

def uniformPlacesProbs(num, pscout):
    probs = np.full(num + 1, (1 - pscout) / num )
    probs[num] = pscout 
    return probs   

class linearPlacesProbs:
    def __init__(self, elitism):
        self.elitism = elitism
    def __call__(self, num, pscout):
        a = self.elitism * (1 - pscout) * 2 / (num ** 2 - num)
        b = (1 - pscout) / num + a * (num - 1) / 2
        probs = -a * np.array(range(num + 1)) + b
        probs[num] = pscout
        return probs
