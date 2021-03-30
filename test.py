import numpy as np
from functools import reduce

def f(**args):
    for a in args.keys():
        print(a, "->", args[a])

def g(**args):
    f(x=3, **args)        

g(a=1, b=2)

