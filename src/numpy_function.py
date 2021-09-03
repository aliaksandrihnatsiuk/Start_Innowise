import numpy as np


def combination(x, y):
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
