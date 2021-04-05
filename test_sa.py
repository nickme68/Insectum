import numpy as np
import insectae as ins

#g = ins.toMax()
#g = ins.toMin()
m = ins.metrics(goal="max", verbose=100)

target = ins.binaryTarget(metrics=m, target=lambda x: np.sum(x), dimension=100)
#target = ins.realTarget(metrics=m, target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
stop = ins.stopMaxGeneration(1000, metrics=m)
#stop = ins.stopValue(98, 1000, metrics=m)

sa = ins.simulatedAnnealing(target=target, goal="max", stop=stop, popSize=20)
sa.theta=ins.hypCool(1.0, 1.0) #expCool(1.0, 0.99)

#sa.opMove = ins.realMutation(delta=ins.hypCool(0.1, 0.07))
sa.opMove = ins.binaryMutation(prob=0.01) #im.expCool(0.1, 0.99))

tm = ins.timer(m)
sa.timer = tm

sa()
m.showTiming()
m.show(log=True) 