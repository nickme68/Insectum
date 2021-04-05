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

    def initAttributes(self, **args):
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
        target.initAttributes(self, encoding="binary", **args)
    def defaultInit(self): 
        return randomBinaryVector()
  
class randomBinaryVector:
    def __call__(self, ind, **xt):
        dim = xt['target'].dimension
        key = xt['key'] 
        ind[key] = np.random.randint(2, size=dim)

# real valued target

class realTarget(target):
    def __init__(self, **args):
        target.__init__(self)
        self.bounds = None 
        target.initAttributes(self, encoding="real", **args)
    def defaultInit(self):
        return randomRealVector()

class randomRealVector:
    def __init__(self, *args):
        self.bounds = None
        if len(args) == 1:
            self.bounds = [-args[0], args[0]]
        elif len(args) == 2:
            self.bounds = [args[0], args[1]]
    def __call__(self, ind, **xt):
        dim = xt['target'].dimension
        low, high = self.bounds if self.bounds != None else xt['target'].bounds
        key = xt['key'] 
        ind[key] = low + (high - low) * np.random.rand(dim)

# permutation task

class permutationTarget(target):
    def __init__(self, args):
        target.__init__(self)
        target.initAttributes(self, encoding="permutation", **args)
    def defaultInit(self):
        return randomPermutation()

class randomPermutation:
    def __call__(self, ind, **xt):
        dim = xt['target'].dimension
        key = xt['key'] 
        ind[key] = np.array(list(range(dim)))
        np.random.shuffle(ind[key])

