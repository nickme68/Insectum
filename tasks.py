import numpy as np 
from alg_base import evalf

def isLarger(x, y):
    return x > y

def isSmaller(x, y):
    return x < y

class task:
    def __init__(self):
        self.encoding = None
        self.target = None 
        self.isBetter = None
        self.dimension = None 

    def initAttributes(self, args):
        self.__dict__.update(args)
 
    def getDir(self):
        if self.isBetter == isLarger:
            return "max"
        return "min"

    def __call__(self, x):
        return self.target(x) 

# binary task
  
class binaryTask(task):
    def __init__(self, args):
        task.__init__(self)
        task.initAttributes(self, args)
    def defaultInit(self):
        return randomBinaryVector()
    @classmethod
    def toMin(cls, **args):
        args.update({"encoding":"binary", "isBetter":isSmaller})
        return cls(args)
    @classmethod
    def toMax(cls, **args):
        args.update({"encoding":"binary", "isBetter":isLarger})
        return cls(args)

class randomBinaryVector:
    def __init__(self):
        pass
    def __call__(self, args):
        dim = args['metrics'].task.dimension
        return np.random.randint(2, size=dim)

# real valued task

class realTask(task):
    def __init__(self, args):
        task.__init__(self)
        self.bounds = None
        task.initAttributes(self, args)
    def defaultInit(self):
        return randomRealVector(self.bounds)
    @classmethod
    def toMin(cls, **args):
        args.update({"encoding":"real", "isBetter":isSmaller})
        return cls(args)
    @classmethod
    def toMax(cls, **args):
        args.update({"encoding":"real", "isBetter":isLarger})
        return cls(args)

class randomRealVector:
    def __init__(self, bounds):
        self.bounds = bounds
    def __call__(self, args):
        dim = args['metrics'].task.dimension
        low, high = evalf(self.bounds, [args])
        return np.random.uniform(low=low, high=high, size=dim)

# permutation task

class permutationTask(task):
    def __init__(self, args):
        task.__init__(self)
        task.initAttributes(self, args)
    def defaultInit(self):
        return randomPermutation()
    @classmethod
    def toMin(cls, **args):
        args.update({"encoding":"permutation", "isBetter":isSmaller})
        return cls(args)
    @classmethod
    def toMax(cls, **args):
        args.update({"encoding":"permutation", "isBetter":isLarger})
        return cls(args)

class randomPermutation:
    def __init__(self):
        pass
    def __call__(self, args):
        dim = args['metrics'].task.dimension
        x = np.array(list(range(dim)))
        np.random.shuffle(x)
        return x

# tests

if __name__ == "__main__":
    t = realTask.toMax(target=lambda x: np.sum(np.square(x)), dimension=10, bounds=[-1, 1])
    print(t.isBetter(11, 2))
    q = t.defaultInit()
    print(q(10))
    print("ok")

    name, q = "A", 2
    print(f"Hello, {name}! How's it {q}?")