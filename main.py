import numpy as np
from src.numpy_function import combination, generation_points, line, coeff
import matplotlib.pyplot as plt

x_min = -1
x_max = 1
N = 300
k = -0.5
b = 3

(x, y) = generation_points(-1,1,-0.5,2,300)
(k_new, b_new) = coeff(x,y)


print ("k =", k, "b =",b)
print ("k_new =", round(k_new,2), "b_new =",round(b_new,2))
plt.scatter(x, y)
plt.plot(x, k*x + b,'r')
plt.plot(x, k_new*x + b_new,'g')
plt.show()
