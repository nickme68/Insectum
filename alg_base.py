import numpy as np 
from random import random, choices
import copy
from patterns import evaluate, neighbors, pairs
from targets import getGoal 

def evalf(param, **xt):
    if callable(param):
        return param(**xt)
    return param

class algorithm: 
    def __init__(self):
        self.target = None
        self.goal = "min"
        self.stop = lambda x: None
        self.popSize = None
        self.opInit = None
        self.env = None
        self.population = None
        self.additionalProcedures = {'start':[],'enter':[], 'exit':[], 'finish':[]}
        self.decorators = []

    def initAttributes(self, **args):
        self.__dict__.update(args)

    def addProcedure(self, key, proc):
        self.additionalProcedures[key].append(proc)

    def runAdds(self, key):
        for proc in self.additionalProcedures[key]:
            proc(self.population, **self.env)        

    def checkKey(self, k):
        if k[0] in '*&':
            d = {'*':'_f', '&':'_x'}
            self.env[d[k[0]]] = k[1:]
            return k[1:]
        return k

    def start(self, envAttrs="", indAttrs="", shadows = ""):
        self.goal = getGoal(self.goal)
        # environment
        keys = ["target", "goal", "time", "timer", "popSize", "_x", "_f"] + envAttrs.split()
        self.env = dict(zip(keys, [None] * len(keys)))
        for key in keys:
            if key in self.__dict__:
                self.env[key] = self.__dict__[key]
        self.env['time'] = 0
        # population
        keys = list(map(self.checkKey, indAttrs.split()))
        ind = dict(zip(keys, [None] * len(keys)))
        self.population = [ind.copy() for i in range(self.popSize)]
        if self.opInit == None:
            self.opInit = self.target.defaultInit()
        # shadow populations
        for sh in shadows.split():
            self.__dict__[sh] = [ind.copy() for i in range(self.popSize)]
        self.runAdds('start')

    def enter(self): 
        self.env['time'] += 1
        self.runAdds('enter')

    def runGeneration(self): pass

    def exit(self): 
        self.runAdds('exit')

    def finish(self): 
        self.runAdds('finish')

    def run(self):
        self.start()
        while not self.stop(self.env):
            self.enter()
            self.runGeneration()
            self.exit()
        self.finish()        

# Decorators

def decorate(obj, D):
    if not isinstance(D, list):
        D = [D]
    for d in D:
        d(obj)

class timeIt:
    def __init__(self, timer):
        self.timer = timer
    def __call__(self, alg):
        if "timeIt" in alg.decorators:
            return
        def start(population, **xt):
            alg.env['timer'] = self.timer
            self.timer.startGlobal()
        def finish(population, **xt):
            self.timer.stopGlobal()
        alg.addProcedure('start', start)
        alg.addProcedure('finish', finish)
        alg.decorators.append("timeIt")

class rankIt:
    def __call__(self, alg):
        if "rankIt" in alg.decorators:
            return
        def start(population, **xt):
            for i in range(len(population)):
                population[i]["_rank"] = i 
        def enter(population, **xt):
            keyf = xt['_f']
            P = list(map(lambda x: x['_rank'], population))
            if xt['goal'] == "min":
                key = lambda x: population[x][keyf]
            else:
                key = lambda x: -population[x][keyf]
            P.sort(key=key)
            for i in range(len(population)):
                population[P[i]]['_rank'] = i
        alg.addProcedure('start', start)
        alg.addProcedure('enter', enter)
        alg.decorators.append("rankIt")

class addElite:
    def __init__(self, size=1):
        self.size = size
    def __call__(self, alg):
        if "addElite" in alg.decorators:
            return
        def enter(population, **xt):
            keyf = xt['_f']
            keyx = xt['_x']
            alg._elite = {}
            n = len(population)
            if isinstance(self.size, float):
                self.size = int(self.size * n)
            P = list(range(n))
            for i in range(self.size):
                for j in range(i + 1, n):
                    if xt['goal'].isBetter(population[P[j]][keyf], population[P[i]][keyf]):
                        P[i], P[j] = P[j], P[i]
                alg._elite[P[i]] = {keyf:population[P[i]][keyf], keyx:population[P[i]][keyx].copy()}
        def exit(population, **xt):
            keyf = xt['_f']
            keyx = xt['_x']
            for i in alg._elite:
                if xt['goal'].isBetter(alg._elite[i][keyf], population[i][keyf]):
                    population[i][keyf] = alg._elite[i][keyf] 
                    population[i][keyx] = alg._elite[i][keyx].copy() 
        alg.addProcedure('enter', enter)
        alg.addProcedure('exit', exit)
        alg.decorators.append("addElite")

# Common stuff

class fillAttribute:
    def __init__(self, op):
        self.op = op
    def __call__(self, ind, **xt):
        key = xt['key']
        if callable(self.op):
            ind[key] = self.op(**xt)
        elif np.isscalar(self.op):
            ind[key] = self.op
        else:
            ind[key] = self.op.copy()

def copyAttribute(ind, **xt):
    keyFrom = xt['keyFrom']
    keyTo = xt['keyTo']
    if np.isscalar(ind[keyFrom]):
        ind[keyTo] = ind[keyFrom]
    else:
        ind[keyTo] = ind[keyFrom].copy()

class mixture:
    def __init__(self, methods, probs):
        self.methods = methods + [None]
        self.probs = probs + [1 - np.sum(probs)]
    def __call__(self, inds, **xt):
        m = choices(self.methods, weights=self.probs)[0]
        if m:
            m(inds, **xt)

class probOp:
    def __init__(self, method, prob):
        self.method = method
        self.prob = prob
    def __call__(self, inds, **xt):
        prob = evalf(self.prob, inds=inds, **xt)
        if random() < prob:
            self.method(inds, **xt)

class timedOp:
    def __init__(self, method, dt):
        self.method = method
        self.dt = dt
    def __call__(self, inds, **xt):
        t = xt['time']
        if t % self.dt == 0:
            self.method(inds, **xt)

class shuffled:
    def __init__(self, op):
        self.op = op
    def __call__(self, population, **xt):
        P = list(range(len(population)))
        np.random.shuffle(P)
        neighbors(population, self.op, P, **xt)

class selected:
    def __init__(self, op):
        self.op = op
    def __call__(self, population, **xt):
        shadow = []
        for i in range(len(population)):
            j = samplex(len(population), 1, [i])[0]
            shadow.append(copy.deepcopy(population[j]))
        pairs(population, shadow, self.op, **xt)

def samplex(n, m, x):
    s = list(set(range(n)) - set(x))
    return list(np.random.choice(s, m, False))

def simpleMove(ind, **xt):
    keyx = xt['keyx']
    keyv = xt['keyv']
    dt = xt['dt']
    ind[keyx] += dt * ind[keyv]


