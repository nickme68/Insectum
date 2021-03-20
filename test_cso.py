import numpy as np
import insectae as ins

g = ins.toMin()
m = ins.metrics(goal=g, verbose=50)
t = ins.realTarget(metrics=m, target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
s = ins.stopMaxGeneration(500, metrics=m)

cso = ins.competitiveSwarmOptimizer(target=t, goal=g, stop=s, popSize=20, delta=0.01, socialFactor=0.1)

cso()
m.show(log=True)