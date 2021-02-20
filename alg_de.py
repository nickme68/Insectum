from alg_base import * 

class differentialEvolution(algorithm):
    def __init__(self, **args):
        algorithm.__init__(self)
        self.opMakeProbes = None
        self.opCorssover = None
        self.opSelect = None
        self.weight = None
        self.xProb = None
        algorithm.initAttributes(self, args)

    def start(self):
        algorithm.start(self, "weight xProb", "x f", shadows="probes")
        foreach(self.population, self.opInit, self.args(key='x'))
        self.evaluateAll()

    def __call__(self):
        self.start()
        while not self.metrics.stopIt():
            self.newGeneration()
            self.opMakeProbes(self.probes, self.population, self.args(key='x'))
            self.opCrossover(self.probes, self.population, self.args(key='x', twoway=False))
            evaluate(self.probes, self.args(keyx='x', keyf='f'))
            self.opSelect(self.population, self.probes, self.args(key='f', twoway=False))
