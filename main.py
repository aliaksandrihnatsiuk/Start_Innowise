import numpy as np
from src.numpy_function import combination

y = np.arange(5, 10, 0.8)
x = np.arange(0, 5, 0.9)

d = combination(x, y)

print(d)