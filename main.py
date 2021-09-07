import numpy as np
from src.numpy_function import combination, generation_points, line, coeff_3,new_coeff
import matplotlib.pyplot as plt

N = 10

w0 = 1
w1 = 1
w2 = 0.5
W = np.array([w0, w1, w2]).reshape(3,1)

x0 = np.ones(N)
x1 = np.linspace(1,2,N)
x2= np.linspace(-1,2,N)

X = np.array([x0, x1, x2]).T
Y = w0 + w1*x1 + x2*w2 + + np.random.normal(0,1,N)
print ('Coefficients, w0, w1, w2')
print(coeff_3(X, Y))



