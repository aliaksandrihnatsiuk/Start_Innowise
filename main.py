import numpy as np
from src.numpy_function import combination, generation_points, line, coeff_3,new_coeff
import matplotlib.pyplot as plt



if __name__ == '__main__':
    N = 50
    w0 = 0.2
    w1 = 0.2
    w2 = 0.5

    def W_creation(w0, w1, w2):
        return np.array([w0, w1, w2]).reshape(3, 1)

    def X_creation (N):
        x0 = np.ones(N*N)
        X1 = np.linspace(1,2,N) + np.random.normal(0, 0.1, N)
        X2 = np.linspace(-1,2,N) + np.random.normal(0, 6, N)
        X = np.array(np.meshgrid(X1,X2)).T.reshape(-1,2)
        X = np.hstack([x0[:,np.newaxis], X])
        return X

    def Y_creation(X, W):
        Y_data = X.dot(W).reshape(N * N) + np.random.normal(0, 2, N * N)
        return Y_data

    def z_function(X, W):
        return np.sum(X*W.T, axis=1)


    W = W_creation(w0, w1, w2)
    X = X_creation(N)
    Y_data = Y_creation(X,W)

    print('Coefficients, w0, w1, w2')
    print(coeff_3(X, Y_data))

    # Draw Plot3D
    Z = z_function(X,W)
    ax = plt.axes(projection='3d')
    ax.scatter(X[:,1],X[:,2], Z[:,np.newaxis])
    ax.scatter(X[:, 1], X[:, 2], Y_data[:, np.newaxis]) #after change on plot surface

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.show()




