import numpy as np
import insectae as ins

m = ins.metrics(verbose=200)
t = ins.realTarget(metrics=m, target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
s = ins.stopMaxGeneration(1000, metrics=m)

cso = ins.competitiveSwarmOptimizer(target=t, stop=s, popSize=20, delta=0.01)
cso.socialFactor = 0.1

cso()
m.show(log=True)