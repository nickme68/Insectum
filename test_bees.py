import numpy as np
import insectae as ins

g = ins.toMin()
m = ins.metrics(goal=g, verbose=200)
t = ins.realTarget(metrics=m, target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
s = ins.stopMaxGeneration(1000, metrics=m)

bees = ins.beesAlgorithm(target=t, goal=g, stop=s, popSize=20, plNum=10, probScout=0.0001)

#bees.opPlaceProbs = ins.uniformPlacesProbs
bees.opPlaceProbs = ins.linearPlacesProbs(0.9)

loc = ins.realMutation(ins.expCool(1, 0.99))
glob = ins.randomRealVector()
bees.opFlight = ins.beeFlight(loc, glob) 

tm = ins.timer(m)
bees.timer = tm

bees()

m.showTiming()
m.show(log=True)