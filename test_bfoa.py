import numpy as np
import insectum as im

target = im.realTask.toMin(target=lambda x: np.sum(np.square(x)), dimension=2, bounds=[-10, 10])
stop = im.stopMaxGeneration(1500)
#stop = im.stopValue(0.000001, 1000)
m = im.metrics(target, stop, verbose=50)

#ab = lambda x: (np.random.random() * 0.1, np.random.random() * 0.1)
bfoa = im.bacterialForagingAlgorithm(metrics=m, popSize=20, gamma=10, probRotate=(0.1, 0.9), vel=im.hypCool(1.0, 1.0), mu=0.001)

bfoa.opSelect = im.shuffled(im.probOp(im.tournament(pwin=1.0),0.5))
bfoa.opDisperse = im.probOp(im.fillAttribute(im.randomRealVector(target.bounds)), 0.0001) 
#bfoa.opDisperse = im.realMutation(im.expCool(0.5, 0.9)) 
#im.realMutation(prob=0.001) #im.expCool(0.1, 0.99))

bfoa.opSignal = im.calcSignals(1, im.signalClustering(0.0001, "min"), "sum")

bfoa()
m.show(log=True)

