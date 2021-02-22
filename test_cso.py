import numpy as np
import insectum as im

target = im.realTask.toMin(target=lambda x: np.sum(np.square(x - 5.0)), dimension=10, bounds=[-10, 10])
stop = im.stopMaxGeneration(500)
#stop = im.stopValue(0.000001, 1000)
m = im.metrics(target, stop, verbose=50)

cso = im.competitiveSwarmOptimizer(metrics=m, popSize=20, delta=0.01, socialFactor=0.1)

cso()
m.show(log=True)
