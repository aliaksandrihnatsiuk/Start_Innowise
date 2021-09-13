import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def E(y, a, b): # function sum eror (f(x) - ax - b)^2
    ff = np.array([a * z + b for z in range(N)])
    return np.dot((y - ff).T, (y-ff))

def dEda(y, a, b):
    ff = np.array([a * z + b for z in range(N)])
    return -2*np.dot((y - ff).T, range(N))

def dEdb(y, a, b):
    ff = np.array([a * z + b for z in range(N)])
    return -2*(y - ff).sum()

N = 50
N_iter = 100 # iteration
sigma = 1 #std
at = 0.5  #parametr k
bt = 0.5   #parametr b

aa = 0 #start value for at(k) - we find it
bb = 0 #start value for bt(b - we find it
lm1 = 0.000001
lm2 = 0.0001

f = np.array([at * z + bt for z in range(N)])
y = np.array(f + np.random.normal(0, sigma, N))

a_plt = np.arange(-1, 2, 0.1)
b_plt = np.arange(0, 3, 0.1)
E_plt = np.array([[E(y, a, b) for a in a_plt] for b in b_plt])

plt.ion()  # Enable interactive mode
fig = plt.figure()
ax = Axes3D(fig)

a, b = np.meshgrid(a_plt, b_plt)
ax.plot_surface(a, b, E_plt, color='y', alpha=0.5)

ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('E')
point = ax.scatter(aa, bb, E(y, aa, bb), c='red')

for n in range(N_iter):
    aa = aa - lm1 * dEda(y, aa, bb)
    bb = bb - lm2 * dEdb(y, aa, bb)
    ax.scatter(aa, bb, E(y, aa, bb), c='red')

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)

    print (aa, bb)
# point = ax.scatter(aa, bb, E(y, aa, bb), c='red')

plt.ioff()
plt.show()

ff = np.array([aa*z+bb for z in range(N)])

plt.scatter(range(N),y, s=2, c='r')
plt.plot(f)
plt.plot(ff, c='red')
plt.grid(True)
plt.show()