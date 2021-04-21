import numpy as np
import insectae as ins

"""
g = ins.toMin()
m = ins.metrics(goal=g, verbose=200)
t = ins.realTarget(metrics=m, target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
s = ins.stopMaxGeneration(1000, metrics=m)
"""

m = ins.metrics(goal="max", verbose=200)
t = ins.binaryTarget(metrics=m, target=lambda x: np.sum(x), dimension=100)
s = ins.stopMaxGeneration(1000, metrics=m)

bees = ins.beesAlgorithm(target=t, goal="max", stop=s, popSize=20, beesNum=20)
#bees.opProbs = ins.uniformPlacesProbs(pscout=0.1)
#bees.opProbs = ins.linearPlacesProbs(0.9, pscout=0.1)
bees.opProbs = ins.binaryPlacesProbs(0.5, 0.9, pscout=0.1)

#bees.opLocal = ins.realMutation(ins.expCool(1, 0.99))
#bees.opGlobal = ins.randomRealVector()

bees.opLocal = ins.binaryMutation(ins.expCool(0.1, 0.99))
bees.opGlobal = ins.randomBinaryVector()

ins.decorate(bees, ins.timeIt(ins.timer(m)))

bees.run()

m.showTiming()
m.show()