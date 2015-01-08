import matplotlib.pyplot as plt
from numpy import *
from dec.spectral2 import Grid_2D_Periodic

X = linspace(0, 2*pi, 30)
Y = linspace(0, 2*pi, 30)
X, Y = meshgrid(X, Y)

g = Grid_2D_Periodic(5, 5)

plt.scatter(*g.verts)

B0, B1, B2, B0d, B1d, B2d = g.basis_fn()
U, V = B1[0][2, 2](X, Y)

plt.quiver(X, Y, U, V)

plt.show()