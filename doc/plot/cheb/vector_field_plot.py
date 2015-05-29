import matplotlib.pyplot as plt
from numpy import *
from dec.grid2 import Grid_2D

X = linspace(0, 2*pi, 30)
Y = linspace(0, 2*pi, 30)
X, Y = meshgrid(X, Y)

g = Grid_2D.periodic(5, 5)

plt.scatter(*g.verts)

B = g.dec.B
U, V = B[1, True](12)(X, Y)

plt.quiver(X, Y, U, V)

plt.show()