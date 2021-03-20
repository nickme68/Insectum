class stopMaxGeneration:
    def __init__(self, maxGen, metrics):
        self.metrics = metrics
        self.maxGen = maxGen
    def __call__(self, env):
        self.metrics.newGeneration()
        return self.metrics.currentGeneration > self.maxGen

class stopMaxEF:
    def __init__(self, maxEF, metrics):
        self.metrics = metrics
        self.maxEF = maxEF
    def __call__(self, env):
        self.metrics.newGeneration()
        return self.metrics.efs > self.maxEF

class stopValue:
    def __init__(self, value, maxGen, metrics):
        self.metrics = metrics
        self.value = value
        self.maxGen = maxGen
    def __call__(self, env):
        self.metrics.newGeneration()
        goal = env['goal']
        a = goal.isBetter(self.value, self.metrics.bestValue)
        b = self.metrics.currentGeneration >= self.maxGen
        return not a or b
