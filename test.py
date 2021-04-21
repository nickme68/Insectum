import numpy as np
from functools import reduce

"""class A:
    def __init__(self):
        self.x = 1
        self.P = lambda: None
    def __call__(self):
        self.P()
        for k in self.__dict__:
            print(k, self.__dict__[k])

def decorate(obj, D):
    if isinstance(D, list): 
        for d in D:
            d(obj)
    else:
        D(obj)
    return obj

class decorA:
    def __init__(self, t):
        self.text = t
    def __call__(self, obj):
        obj.a = self.text
        return obj

def decorB(obj):
    obj.b = [1,2,3]
    return obj

class decorC:
    def __init__(self):
        pass
    def __call__(self, obj):
        def proc():
            obj.x = 444
            obj.y = "Hi"
        obj.P = proc

a = decorate(A(), [decorA("Hello!"), decorB])
#a = decorate(A(), decorC())
#a = decorB(A())
a()

"""

d = {'a':1, 'v':22}

if 'a' in d: print(d['a'])
if 'b' in d: print(d['b'])


