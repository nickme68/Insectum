import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
from targets import getGoal
from time import time

class metrics:
    def __init__(self, goal="min", verbose=0):
        self.bestSolution = None
        self.bestValue = None
        self.efs = 0
        self.currentGeneration = 0
        self.data = [[]]
        self.verbose = verbose
        self.goal = getGoal(goal)
        self.timing = {}
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
            if self.goal == "min":
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
    def format(self, label, L, t, T):
        p = ""
        if T > 0: 
            p = "({:.2f}%)".format(t / T * 100) 
        lab = ("{:" + str(L) + "s} ").format(label)
        return "  " + lab + ": {:6.3f} ".format(t) + p
    def showTiming(self):
        print("[timing]")
        if self.timing == {}:
            print("  no timing information")
            return
        labels = self.timing.keys()
        L = max(map(len, labels))
        T = 0
        if 'total' in labels:
            T = self.timing['total']
        for label in self.timing:
            if label != 'total':
                print(self.format(label, L, self.timing[label], T))
        print("  " + "=" * L)
        if label in labels:
            print(self.format('total', L, T, 0)) #"  " + (label + " " * L)[:L]+" :", str(self.timing[label])[:10])

class timer:
    def __init__(self, metrics):
        self.metrics = metrics
        self.tglobal = None
        self.tlocal = None
    def startGlobal(self):
        self.tglobal = time()
    def stopGlobal(self):
        self.metrics.timing['total'] = time() - self.tglobal
    def startLocal(self):
        self.tlocal = time()
    def stopLocal(self, label):
        if label not in self.metrics.timing:
            self.metrics.timing[label] = 0.0
        self.metrics.timing[label] += time() - self.tlocal

def timing(f):
    def func(*x, **xt):
        doTiming = '_t' in xt and xt['timer'] != None
        if doTiming:
            xt['timer'].startLocal()
        result = f(*x, **xt)
        if doTiming: 
            xt['timer'].stopLocal(xt['_t'])
        return result
    return func

