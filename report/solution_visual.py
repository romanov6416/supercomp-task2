

from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math


# Replace it with your Phi() function.
def phi(x, y):
    t = x + y
    return math.e ** (1 - t * t)


# fig = plt.figure()
# ax = fig.gca(projection='3d')
# z = np.array([])
# x = np.array([])
# y = np.array([])
# ax.plot(x, y, z, label='parametric curve')
# ax.legend()

# plt.show()
N = 2000
A = 0.0
B = 2.0
x = np.array([[i * (B - A) for j in range(N)] for i in range(N)])
y = np.array([[j * (B - A) for j in range(N)] for i in range(N)])
z = np.array([[phi(x[i][j], y[i][j]) for j in range(N)] for i in range(N)])
print("inited")

with open("p_2000.txt") as f:
    lines = f.readlines()
print("file is read")

for s in lines:
    i, j, xij, yij, pij = s.split()
    i, j = int(i), int(j)
    x[i][j] = float(xij)
    y[i][j] = float(yij)
    z[i][j] = float(pij)
print("loaded")

fig = plt.figure(figsize=(10, 10))
# fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
surf = ax.plot_surface(x, y, z)
plt.show()

