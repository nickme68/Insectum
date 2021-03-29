import numpy as np 

class toMax:
    def isBetter(self, x, y):
        return x > y
    def getDir(self):
        return "max"

class toMin:
    def isBetter(self, x, y):
        return x < y
    def getDir(self):
        return "min"

def getGoal(g):
    if g == "min":
        return toMin()
    if g == "max":
        return toMax()
    return g
    
class target:
    def __init__(self):
        self.encoding = None 
        self.target = lambda x: None
        self.dimension = None
        self.metrics = None

    def initAttributes(self, args):
        self.__dict__.update(args)

    def __call__(self, x, f, reEval):
        if reEval:
            f = self.target(x)
        self.metrics.newEval(x, f, reEval) 
        return f

# binary target

class binaryTarget(target):
    def __init__(self, **args):
        target.__init__(self)
        args.update({"encoding":"binary"})
        target.initAttributes(self, args)
    def defaultInit(self): 
        return randomBinaryVector(self.dimension) 
  
class randomBinaryVector:
    def __init__(self, dim):
        self.dim = dim
    def __call__(self, args): 
        return np.random.randint(2, size=self.dim)

# real valued target

class realTarget(target):
    def __init__(self, **args):
        target.__init__(self)
        self.bounds = None 
        args.update({"encoding":"real"})
        target.initAttributes(self, args)
    def defaultInit(self):
        return randomRealVector(self.dimension, self.bounds)

class randomRealVector:
    def __init__(self, dim, bounds):
        self.dim = dim
        self.bounds = bounds
    def __call__(self, args):
        low, high = self.bounds
        return np.random.uniform(low=low, high=high, size=self.dim)

# permutation task

class permutationTarget(target):
    def __init__(self, args):
        target.__init__(self)
        args.update({"encoding":"permutation"})
        target.initAttributes(self, args)
    def defaultInit(self):
        return randomPermutation(self.dimension)

class randomPermutation:
    def __init__(self, dim):
        self.dim = dim
    def __call__(self, args):
        x = np.array(list(range(self.dim)))
        np.random.shuffle(x)
        return x
