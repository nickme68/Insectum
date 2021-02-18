from alg_base import * 
import copy

class simulatedAnnealingAlgorithm(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opMove = None
        self.theta = None
        algorithm.initAttributes(self, args)

    @staticmethod
    def accept(ind, args):
        theta = evalf(args['env']['theta'], [args, ind])
        task = args['metrics'].task
        df = abs(ind['fNew'] - ind['f'])
        if task.isBetter(ind['fNew'], ind['f']) or np.random.random() < np.exp(-df / theta):
            ind['f'] = ind['fNew']
            ind['x'] = ind['xNew'].copy()

    def start(self):
        algorithm.start(self, "theta", "x xNew f fNew")
        foreach(self.population, self.opInit, self.args(key='x'))
        self.evaluateAll()

    def __call__(self):
        self.start()
        while not self.metrics.stopIt():
            self.newGeneration()
            foreach(self.population, copyAttribute, self.args(keyFrom='x', keyTo='xNew'))
            foreach(self.population, self.opMove, self.args(key='xNew'))
            self.evaluateAll(keyx='xNew', keyf='fNew')
            foreach(self.population, self.accept, self.args())

