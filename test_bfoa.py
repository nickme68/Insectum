import numpy as np
import insectae as ins

#g = ins.toMin()
m = ins.metrics(verbose=200)

t = ins.realTarget(metrics=m, target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
s = ins.stopMaxGeneration(1000, metrics=m)

bfoa = ins.bacterialForagingAlgorithm(target=t, stop=s, popSize=40, gamma=0.1, probRotate=(0.01, 0.99), vel=ins.expCool(1.0, 0.99), mu=0.01)

bfoa.opSelect = ins.shuffled(ins.timedOp(ins.probOp(ins.tournament(pwin=0.6), 0.9), 10))

#bfoa.opDisperse = ins.probOp(ins.randomRealVector(), 0.0001) 

bfoa.opDisperse = ins.timedOp(ins.probOp(ins.randomRealVector(), 0.01), 20) 

#bfoa.opSignals = ins.noSignals()
bfoa.opSignals = ins.timedOp(ins.calcSignals(shape=ins.shapeClustering(0.00001)), 10)
#bfoa.opSignals = ins.timedOp(ins.calcSignals(shape=ins.signalClustering(ins.expCool(0.001, 0.999), "min")), 10)

tm = ins.timer(m)
#ins.decorate(bfoa, [ins.timeIt(tm), ins.addElite(1)])
ins.decorate(bfoa, ins.timeIt(tm))

bfoa.run()
#m.showTiming()
m.show(log=True)