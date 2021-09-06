from math import pi, sqrt, e
import numpy as np
import matplotlib.pyplot as plt

def combination(x, y):
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

def line(k, b, x):
    plt.plot(x, k * x + b, 'r')


def generation_points(x_min, x_max, k, b, N):
    x = np.linspace(x_min, x_max, N)
    y = k * x + b + np.random.normal(-1, 2, N)
    plt.scatter(x, y)
    return x,y

def coeff(x,y):
    coeff = np.polyfit(x,y,1)
    k_new = np.polyfit(x,y,1)[0]
    b_new = np.polyfit(x,y,1)[1]
    return k_new, b_new

def new_coeff(x,y,N):
    for i in range(N):
        k = (N * sum(x * y) - sum(x) * sum(y)) / (N * sum(x * x) - (sum(x)) ** 2)
        b = (sum(y) - k * sum(x)) / N
        return k, b