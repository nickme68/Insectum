import numpy as np
import insectae as ins

g = ins.toMin()
m = ins.metrics(goal=g, verbose=200)
#t = ins.binaryTarget(metrics=m, target=lambda x: np.sum(x), dimension=100)
t = ins.realTarget(metrics=m, target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
s = ins.stopMaxGeneration(1000, metrics=m)

ga = ins.geneticAlgorithm(target=t, goal=g, stop=s, popSize=20)

ga.opSelect = ins.shuffled(ins.probOp(ins.tournament(pwin=0.9), 0.5))
#ga.opSelect = ins.selected(ins.probOp(ins.tournament(pwin=0.9), 0.5))
x1 = ins.uniformCrossover(pswap=0.3)
x2 = ins.singlePointCrossover()
x3 = ins.doublePointCrossover()
x = ins.mixture([x1, x2, x3], [0.2, 0.2, 0.2])
ga.opCrossover = ins.shuffled(x2)
#ga.opCrossover = ins.selected(x2) # TODO check it!
#ga.opCrossover = ins.shuffled(ins.timedOp(x1, 10))
#ga.opCrossover = ins.shuffled(ins.probOp(x1, 0.1))
ga.opCrossover = ins.shuffled(x)

ga.opMutate = ins.realMutation(delta=ins.expCool(0.5, 0.99))
#ga.opMutate = ins.realMutation(delta=ins.hypCool(0.1, 0.25))
#ga.opMutate = ins.binaryMutation(prob=ins.expCool(0.1, 0.99)) #ins.expCool(0.1, 0.99))

ga()
m.show(log=True) 