import numpy as np
import insectum as ins

target = ins.realTask.toMin(target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
#target = ins.binaryTask.toMax(target=lambda x: np.sum(x), dimension=100)
stop = ins.stopMaxGeneration(500)
m = ins.metrics(target, stop, verbose=50)

sa = ins.simulatedAnnealingAlgorithm(metrics=m, popSize=20)
sa.theta=ins.expCool(1.0, 0.99)
sa.opMove = ins.realMutation(0.1)
#sa.opMove = ins.binaryMutation(prob=0.01) #im.expCool(0.1, 0.99))

sa()
m.show() 


