import numpy as np
import random
from random import random, randrange, randint
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from sklearn.utils import shuffle


def init_points(start_point, end_point, rect=None):
     while True:
         rect = []
         for i in range(4):
             rect.append(randint(start_point, end_point ))
         if rect[0] != rect[2] and rect[1] != rect[3]:
             return reorder_coordinates(rect)


def init_points_by_coord(rect):
    while True:
        x1 = randint(rect[2]-1, 50)
        x2 = randint(rect[2]-1, 50)
        y1 = randint(0, 50)
        y2 = randint(0, 50)
        rect = [x1, y1, x2, y2]
        if rect[0] != rect[2] and rect[1] != rect[3]:
            return reorder_coordinates(rect)

def reorder_coordinates(rect):
    if rect[0] > rect[2]:
        rect[0], rect[2] = rect[2], rect[0]
    if rect[1] > rect[3]:
        rect[1], rect[3] = rect[3], rect[1]
    return rect


def generate_two_rects():
        rect1 = init_points(0, 40)
        rect2 = init_points_by_coord(rect1)
        return [rect1, rect2]

def draw_img(rect):
    matrix = np.full((50,50),0, dtype="uint8")
    matrix[rect[0][1]:rect[0][3], rect[0][0]:rect[0][2]] = 1
    matrix[rect[1][1]:rect[1][3], rect[1][0]:rect[1][2]] = 2
    return matrix

def generate_X_Y():
    X = []
    Y = []
    for i in range(8000):
        rect = generate_two_rects()
        rect_coord = list(np.array(rect).reshape(1, -1)[0])
        matrix = np.full((50, 50), 0, dtype="uint8")
        matrix[rect[0][1]:rect[0][3], rect[0][0]:rect[0][2]] = 1
        matrix[rect[1][1]:rect[1][3], rect[1][0]:rect[1][2]] = 2
        X_row = matrix.reshape(1,-1)[0]
        X.append(X_row)
        Y.append(rect_coord)
    X = np.array(X)
    Y = np.array(Y)
    return X.T, Y.T

def mini_batch(X, Y, n_batchs):
    n_test = 800
    Y_true =  Y[:, n_test:]
    X_train = X[:, n_test:]

    X_batches = np.array_split(X_train, n_batchs, axis=1)
    Y_batches = np.array_split(Y_true, n_batchs, axis=1)

    return X_batches, Y_batches

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

def init_params():
    W1 = np.random.rand(neyron, 2500) - 0.5
    b1 = np.random.rand(neyron, 1) - 0.5
    W2 = np.random.rand(8, neyron) - 0.5
    b2 = np.random.rand(8, 1) - 0.5
    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = ReLU(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):

    m_batch = Y.shape[1]
    dZ2 = - 2 * (1 / m_batch) * (Y - A2) * ReLU_deriv(Z2)
    dW2 = dZ2 @ A1.T
    db2 = np.sum(dZ2)

    dA1 = W2.T @ dZ2
    dZ1 = dA1 * ReLU_deriv(Z1)
    dW1 = dZ1 @ X.T
    db1 = np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2

def update_params_nesterov(W1, b1, W2, b2, dW1, db1, dW2, db2, change_W1, change_b1, change_W2, change_b2, learning_rate, momentum):
    change_W1 = (momentum * change_W1) - (learning_rate * dW1)
    change_b1 = (momentum * change_b1) - (learning_rate * db1)
    change_W2 = (momentum * change_W2) - (learning_rate * dW2)
    change_b2 = (momentum * change_b2) - (learning_rate * db2)

    W1 = W1 + change_W1
    b1 = b1 + change_b1
    W2 = W2 + change_W2
    b2 = b2 + change_b2

    return W1, b1, W2, b2

def gradient_descent(X, Y, epoch, n_batchs):

    W1, b1, W2, b2 = init_params()
    X_batches, Y_batches = mini_batch(X, Y, 32)

    x_plot_train =[]
    y_plot_train =[]
    for i in range(epoch):

        X_batches, Y_batches = shuffle(X_batches, Y_batches, random_state=n_batchs)



        for k in range(n_batchs):
            m_batch = Y_batches[k].shape[1]
            #print('Y_batches shape', m_batch)
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_batches[k])
            #print(Z1, A1, Z2, A2)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X_batches[k], Y_batches[k])

            # Update with GD
            #W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

            # Update with GD-Nesterov
            W1, b1, W2, b2 = update_params_nesterov(W1, b1, W2, b2, dW1, db1, dW2, db2, change_W1, change_b1, change_W2, change_b2, learning_rate, momentum)

            loss = (1/m_batch)*np.sum((A2 - Y_batches[k])**2)

        if i % 10 == 0:
            print('epochs:', i)
            print('loss', loss)
            print('Number of coincidences', np.sum(A2 == Y_batches))

            print('A2', list(np.around(np.array(A2[:, 0]), 2)))
            print('Y2', Y[:, 0])
        x_plot_train.append(i)
        y_plot_train.append(loss)


    return W1, b1, W2, b2, x_plot_train,  y_plot_train


# Parametrs

neyron = 64
learning_rate =0.001
momentum = 0.9
change_W1 = 0.001
change_b1 = 0.001
change_W2 = 0.001
change_b2 = 0.001


X, Y =  generate_X_Y()
W1, b1, W2, b2, x_plot_train,  y_plot_train = gradient_descent(X, Y, 100, 32)


plt.plot(x_plot_train, y_plot_train, "r", label="los")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()




#Draw Rectangle with Changed Coordinates

# my_rects = generate_two_rects()
# mat = draw_img(my_rects)
# mat1 = mat.reshape(1,-1)[0]
#
# fig, ax = plt.subplots()
# fig.set_size_inches(10, 10)
# column_labels = np.arange(1, 51, 1)
# row_labels = np.arange(1, 51, 1)
# # put the major ticks at the middle of each cell
# ax.set_xticks(np.arange(mat.shape[1]) + 0.5, minor=False)
# ax.set_yticks(np.arange(mat.shape[0]) + 0.5, minor=False)
# # want a more natural, table-like display
# ax.invert_yaxis()
# ax.xaxis.tick_top()
# ax.grid()
# ax.set_xticklabels(column_labels, minor=False)
# ax.set_yticklabels(row_labels, minor=False)
#plt.imshow(mat)
#print('Current coordinate', my_rects)
#plt.show()