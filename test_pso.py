import numpy as np
import insectae as ins

g = ins.toMin()
m = ins.metrics(goal=g, verbose=50)
t = ins.realTarget(metrics=m, target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
s = ins.stopMaxGeneration(500, metrics=m)

pso = ins.particleSwarmOptimization(target=t, goal=g, stop=s, popSize=100, gamma=0.95, alphabeta=(0.1, 0.1), delta=0.01)
#pso.opLimitVel = im.maxAmplitude(0.95)
pso.opLimitVel = ins.maxAmplitude(ins.expCool(0.5, 0.999))

pso()
m.show(log=True)

