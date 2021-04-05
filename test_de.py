import numpy as np
import insectae as ins

g = ins.toMin()
m = ins.metrics(goal=g, verbose=200)
t = ins.realTarget(metrics=m, target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
s = ins.stopMaxGeneration(1000, metrics=m)

de = ins.differentialEvolution(target=t, goal=g, stop=s, popSize=20)

#tm = ins.timer(m)
#de.timer = tm

de.opMakeProbe = ins.probeClassic(0.8) #probesBest2 #probesCur2Best #probesBest #probesClassic
de.opCrossover = ins.uniformCrossover(0.9)
de.opSelect = ins.tournament(1.0)

de()
m.showTiming()
m.show(log=True)