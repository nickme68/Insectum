import numpy as np
from functools import reduce

x = np.ones(10)
print(x)

def f(x):
    if x > 0: return x + 1
    return x - 1

print(f(x))
print(np.array(list(map(f, x))))

