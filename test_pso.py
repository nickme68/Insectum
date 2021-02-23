import numpy as np
import insectum as im

target = im.realTask.toMin(target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
stop = im.stopMaxGeneration(500)
#stop = im.stopValue(0.000001, 1000)
m = im.metrics(target, stop, verbose=50)

#ab = lambda x: (np.random.random() * 0.1, np.random.random() * 0.1)
pso = im.particleSwarmOptimization(metrics=m, popSize=20, gamma=0.95, alphabeta=(0.1, 0.1), delta=0.01)
#pso.opLimitVel = im.maxAmplitude(0.95)
pso.opLimitVel = im.maxAmplitude(im.expCool(1.0, 0.999))

pso()
m.show(log=True)

