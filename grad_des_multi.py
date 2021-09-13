import numpy as np
import time
import matplotlib.pyplot as plt

def data_creation(w0, w1, w2, N):
    W = np.array([w0, w1, w2]).reshape(3, 1)
    x0 = np.ones(N * N)
    X1 = np.linspace(1, 2, N) + np.random.normal(0, 0.1, N)
    X2 = np.linspace(-1, 2, N) + np.random.normal(0, 6, N)
    X = np.array(np.meshgrid(X1, X2)).T.reshape(-1, 2)
    X = np.hstack([x0[:, np.newaxis], X])
    y = X.dot(W).reshape(N * N) + np.random.normal(0, 2, N * N)
    return W, X, y

def cost_function(X, y, theta):
    J = np.sum((X.dot(theta) -y) ** 2) / (2 * m)
    return J

def gradient_descent(X, y, theta, alpha, iterations):
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = X.dot(theta)
        print(predictions.reshape(m,1))
        errors = np.subtract(predictions, y)
        print(errors.reshape(m,1))
        theta = theta - (alpha / m) * X.transpose().dot(errors)
        cost_history[i] = cost_function(X, y, theta)
        return theta, cost_history




W, X, y = data_creation(0.6, 0.3, 1, 3)


theta = np.array([0.6, 0.2, 1])
# start parameters
iterations = 400
alpha = 0.001
m = len(y)

print(m)

theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

J = cost_function(X, y, theta)
print (J)
print('Final value of theta =', theta)
plt.plot(range(1, iterations +1), cost_history, color ='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel("iterations")
plt.ylabel("cost (J)")
plt.show()
