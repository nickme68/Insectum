import numpy as np
import insectum as im

target = im.realTask.toMin(target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
stop = im.stopMaxGeneration(500)
m = im.metrics(target, stop, verbose=50)

de = im.differentialEvolution(metrics=m, popSize=20, weight=0.8)

de.opMakeProbes = im.probesBest #probesClassic
de.opCrossover = im.uniformCrossover(0.9)
de.opSelect = im.tournament(1.0)

de()
m.show()