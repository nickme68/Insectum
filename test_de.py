import numpy as np
import insectum as im

#target = im.realTask.toMin(target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
target = im.binaryTask.toMax(target=lambda x: np.sum(x), dimension=100)
stop = im.stopMaxGeneration(10)
m = im.metrics(target, stop, verbose=5)

de = im.differentialEvolution(metrics=m, popSize=20, weight=0.8, xProb=0.9)

de()
#m.show()