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
    def newEval(self, x, f, reEval):
        if reEval:
            f = self.task(x)
            self.efs += 1
            if self.bestValue == None or self.task.isBetter(f, self.bestValue):
                self.bestValue = f
                self.bestSolution = np.copy(x)
        self.data[-1].append(f)
        return f
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
        ax = plt.subplots(figsize=(width, height))[1]
        ax.plot(X, B, color="green", label="best")
        ax.plot(X, A, color="blue", label="average")
        ax.plot(X, M, color="red", label="median")
        ax.legend()
        if log:
            ax.set_yscale('log')
        plt.show()

# classes for stopping main loop in all algorithms
 
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