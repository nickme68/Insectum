import numpy as np
import insectum as im

#target = im.realTask.toMin(target=lambda x: np.sum(np.square(x)), dimension=2, bounds=[-10, 10])
target = im.binaryTask.toMin(target=lambda x: np.sum(x), dimension=100)
stop = im.stopMaxGeneration(500)
#stop = im.stopValue(0.000001, 1000)
m = im.metrics(target, stop, verbose=50)

bees = im.beesAlgorithm(metrics=m, popSize=20, plNum=10, probScout=0.01)

bees.opPlaceProbs = im.uniformPlacesProbs

#loc = im.realMutation(im.expCool(0.1, 0.99))
#glob = im.fillAttribute(im.randomRealVector(target.bounds))
loc = im.binaryMutation(im.expCool(0.1, 0.99))
glob = im.fillAttribute(im.randomBinaryVector())
bees.opFlight = im.beeFlight(loc, glob) 

bees()
m.show(log=True)