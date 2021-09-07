import numpy as np
from src.numpy_function import combination, generation_points, line, coeff_3,new_coeff
import matplotlib.pyplot as plt



if __name__ == '__main__':
    N = 15
    w0 = 2
    w1 = 0.1
    w2 = 0.5
    W = np.array([w0, w1, w2]).reshape(3,1)

    x0 = np.ones(N)
    x1 = np.linspace(1,2,N)
    x2= np.linspace(-1,2,N)

    X = np.array([x0, x1, x2]).T
    Y = w0 + w1*x1 + x2*w2 + np.random.normal(0,1,N)
    print ('Coefficients, w0, w1, w2')
    print(coeff_3(X, Y))
    # Draw Plot3D

    def y_function(x1, x2):
        return w0 + w1 * x1 + x2 * w2

    X1, X2 = np.meshgrid(x1, x2)
    Y = y_function(X1, X2)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X1, X2, Y)
    ax.scatter(X1, X2, Y, c='r')
    plt.show()




