import numpy as np
import insectae as ins

m = ins.metrics(verbose=200)
#t = ins.binaryTarget(metrics=m, target=lambda x: np.sum(x), dimension=100)
t = ins.realTarget(metrics=m, target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
s = ins.stopMaxGeneration(1000, metrics=m)

ga = ins.geneticAlgorithm(target=t, stop=s, popSize=40)

ga.opSelect = ins.shuffled(ins.probOp(ins.tournament(pwin=0.9), 0.5))
#ga.opSelect = ins.rankSelection(bounds=(1,2), alg=ga)
#ga.opSelect = ins.selected(ins.probOp(ins.tournament(pwin=0.9), 0.5))
x1 = ins.uniformCrossover(pswap=0.3)
x2 = ins.singlePointCrossover()
x3 = ins.doublePointCrossover()
x = ins.mixture([x1, x2, x3], [0.2, 0.2, 0.2])
ga.opCrossover = ins.shuffled(x)
#ga.opCrossover = ins.selected(x2) # TODO check it!
#ga.opCrossover = ins.shuffled(ins.timedOp(x1, 10))
#ga.opCrossover = ins.shuffled(ins.probOp(x1, 0.1))
ga.opCrossover = ins.shuffled(x)

class binaryRanks:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self, **xt):
        r = xt['inds'][0]['_rank']
        ps = xt['popSize']
        if r / ps < 0.5: 
            return self.a 
        return self.b
        #return self.a + (self.b - self.a) * (r + 1) / ps

class expRanks:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self, **xt):
        r = xt['inds'][0]['_rank']
        ps = xt['popSize']
        return self.a * (self.b / self.a) ** (r / (ps - 1))

ga.opMutate = ins.realMutation(delta=expRanks(0.0000000000001, 1))

#ga.opMutate = ins.realMutation(delta=ins.expCool(0.5, 0.99))
#ga.opMutate = ins.realMutation(delta=ins.hypCool(0.1, 0.25))
#ga.opMutate = ins.binaryMutation(prob=ins.expCool(0.1, 0.99)) #ins.expCool(0.1, 0.99))

#ins.decorate(ga, ins.rankIt())

ins.decorate(ga, [ins.addElite(0.2), ins.timeIt(ins.timer(m)), ins.rankIt()]) 

ga.run()

#m.showTiming()
m.show(log=True) 