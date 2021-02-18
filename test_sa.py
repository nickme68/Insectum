import numpy as np
import insectum as im

#target = im.realTask.toMin(target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
target = im.binaryTask.toMax(target=lambda x: np.sum(x), dimension=100)
stop = im.stopMaxGeneration(500)
m = im.metrics(target, stop, verbose=50)

sa = im.simulatedAnnealingAlgorithm(metrics=m, popSize=20, theta=im.expCool(1.0, 0.99))
#sa.opMove = im.realMutation(0.1)

sa.opMove = im.binaryMutation(prob=0.01) #im.expCool(0.1, 0.99))

sa()
m.show(log=True)

