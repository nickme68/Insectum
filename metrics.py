import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 

class metrics:
    def __init__(self, task, stop, verbose=0):
        self.task = task
        self.stop = stop
        self.bestSolution = None
        self.bestValue = None
        self.efs = 0
        self.currentGeneration = 0
        self.data = [[]]
        self.verbose = verbose
    def newEval(self, ind, keyx, keyf, reEval):
        if reEval:
            ind[keyf] = self.task(ind[keyx])
            self.efs += 1
            if self.bestValue == None or self.task.isBetter(ind[keyf], self.bestValue):
                self.bestValue = ind[keyf]
                self.bestSolution = np.copy(ind[keyx])
        self.data[-1].append(ind[keyf])
    def newGeneration(self):
        self.currentGeneration += 1
        if self.verbose > 0 and self.currentGeneration % self.verbose == 0:
            print(f"Generation: {self.currentGeneration}, EFs: {self.efs}, target: {self.bestValue}")
        self.data.append([])
    def stopIt(self):
        return self.stop(self)
    def show(self, width=8, height=6, log=False):
        B, A, M = [], [], []
        for rec in self.data:
            if rec == []:
                continue 
            x = np.array(rec)
            if self.task.getDir() == "min":
                B.append(np.min(x))
            else:
                B.append(np.max(x))
            A.append(np.mean(x))
            M.append(np.median(x))
        X = list(range(len(B)))
        fig, ax = plt.subplots(figsize=(width, height))
        ax.plot(X, B, color="green", label="best")
        ax.plot(X, A, color="blue", label="average")
        ax.plot(X, M, color="red", label="median")
        ax.legend()
        if log:
            ax.set_yscale('log')
        plt.show()

class stopMaxGeneration:
    def __init__(self, maxGen):
        self.maxGen = maxGen
    def __call__(self, metrics):
        return metrics.currentGeneration >= self.maxGen

class stopMaxEF:
    def __init__(self, maxEF):
        self.maxEF = maxEF
    def __call__(self, metrics):
        return metrics.efs >= self.maxEF

class stopValue:
    def __init__(self, value, maxGen):
        self.value = value
        self.maxGen = maxGen
    def __call__(self, metrics):
        return not metrics.task.isBetter(self.value, metrics.bestValue) or  metrics.currentGeneration >= self.maxGen