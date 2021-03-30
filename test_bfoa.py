import numpy as np
import insectae as ins

g = ins.toMin()
m = ins.metrics(goal=g, verbose=50)
t = ins.realTarget(metrics=m, target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
s = ins.stopMaxGeneration(500, metrics=m)

bfoa = ins.bacterialForagingAlgorithm(goal=g, target=t, stop=s, popSize=100, gamma=0.1, probRotate=(0.01, 0.99), vel=ins.expCool(1.0, 0.99), mu=0.01)

bfoa.opSelect = ins.shuffled(ins.timedOp(ins.probOp(ins.tournament(pwin=0.6), 0.9), 10))
bfoa.opDisperse = ins.probOp(ins.fillAttribute(ins.randomRealVector(t.dimension, t.bounds)), 0.0001) 
#bfoa.opDisperse = ins.timedOp(ins.probOp(ins.realMutation(ins.expCool(0.001,0.999)), 0.1), 20) 
#im.realMutation(prob=0.001) #im.expCool(0.1, 0.99))

#bfoa.opSignals = ins.noSignals()
bfoa.opSignals = ins.timedOp(ins.calcSignals(shape=ins.signalClustering(0.00001, "min")), 10)
#bfoa.opSignals = ins.timedOp(ins.calcSignals(shape=ins.signalClustering(ins.expCool(0.001, 0.999), "min")), 10)

bfoa()
m.show(log=True)