import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 

class metrics:
    def __init__(self, goal, verbose=0):
        self.bestSolution = None
        self.bestValue = None
        self.efs = 0
        self.currentGeneration = 0
        self.data = [[]]
        self.verbose = verbose
        self.goal = goal
    def newEval(self, x, f, reEval):
        if reEval:
            self.efs += 1
            if self.bestValue == None or self.goal.isBetter(f, self.bestValue):
                self.bestValue = f
                self.bestSolution = np.copy(x)
        self.data[-1].append(f)
    def newGeneration(self):
        if self.verbose > 0 and self.currentGeneration % self.verbose == 0:
            g, e, b = self.currentGeneration, self.efs, self.bestValue
            print(f"Generation: {g}, EFs: {e}, target: {b}")
        self.currentGeneration += 1
        self.data.append([])
    def show(self, width=8, height=6, log=False):
        B, A, M = [], [], []
        for rec in self.data:
            if rec == []:
                continue 
            x = np.array(rec)
            if self.goal.getDir() == "min":
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
