from numpy import *
from dec.grid2 import Grid_2D_Periodic
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

X = linspace(0, 2*pi, 100)
Y = linspace(0, 2*pi, 100)
X, Y = meshgrid(X, Y)
g = Grid_2D_Periodic(7, 7)
B0, B1, B2, B0d, B1d, B2d = g.basis_fn()

fig = plt.figure()
ax = fig.gca(projection='3d')
for i, j, c in [(0, 0, 'r'), 
                (2, 2, 'g'), 
                (4, 4, 'b')]:
    Z = B0[i, j](X,Y)
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, color=c, alpha=0.4)

fig = plt.figure()
ax = fig.gca(projection='3d')
for i, j, c in [(0, 0, 'r'), 
                (2, 2, 'g'), 
                (4, 4, 'b')]:
    Z = B0d[i, j](X,Y)
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, color=c, alpha=0.4)

fig = plt.figure()
ax = fig.gca(projection='3d')
for i, j, c in [(0, 0, 'r'), 
                (2, 2, 'g'), 
                (4, 4, 'b')]:
    Z = B2[i, j](X,Y)
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, color=c, alpha=0.4)

fig = plt.figure()
ax = fig.gca(projection='3d')
for i, j, c in [(0, 0, 'r'), 
                (2, 2, 'g'), 
                (4, 4, 'b')]:
    Z = B2d[i, j](X,Y)
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, color=c, alpha=0.4)

plt.show()
