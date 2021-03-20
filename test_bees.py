import numpy as np
import insectae as ins

g = ins.toMin()
m = ins.metrics(goal=g, verbose=200)
t = ins.realTarget(metrics=m, target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
s = ins.stopMaxGeneration(1000, metrics=m)

bees = ins.beesAlgorithm(target=t, goal=g, stop=s, popSize=20, plNum=10, probScout=0.0001)

#bees.opPlaceProbs = im.uniformPlacesProbs
bees.opPlaceProbs = ins.linearPlacesProbs(0.9)

loc = ins.realMutation(ins.expCool(1, 0.99))
glob = ins.fillAttribute(ins.randomRealVector(t.dimension, t.bounds))
#loc = im.binaryMutation(im.expCool(0.1, 0.99))
#glob = im.fillAttribute(im.randomBinaryVector())
bees.opFlight = ins.beeFlight(loc, glob) 

bees()
m.show(log=True)