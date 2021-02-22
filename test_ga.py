import numpy as np
import insectum as im

#target = im.realTask.toMin(target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-10, 10])
target = im.binaryTask.toMax(target=lambda x: np.sum(x), dimension=100)
stop = im.stopMaxGeneration(500)
m = im.metrics(target, stop, verbose=50)

ga = im.geneticAlgorithm(metrics=m, popSize=20)

ga.opSelect = im.shuffled(im.probOp(im.tournament(pwin=0.9), 0.5))
x1 = im.uniformCrossover(pswap=0.3)
x2 = im.singlePointCrossover()
x3 = im.doublePointCrossover()
x = im.probOp(x1, 0.5) #im.mixture([x1, x2, x3], [0.2, 0.2, 0.2])
ga.opCrossover = im.shuffled(x2)
#ga.opCrossover = im.selected(x2) # TODO check it!

# im.mixture(x1, 0.2)

#ga.opMutate = im.realMutation(delta=im.expCool(0.5, 0.99))
#ga.opMutate = im.realMutation(delta=im.hypCool(0.1, 0.25))
ga.opMutate = im.binaryMutation(prob=0.001) #im.expCool(0.1, 0.99))

ga()
m.show()

