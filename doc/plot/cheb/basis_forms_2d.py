import numpy as np
from dec.grid2 import Grid_2D, basis_fn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

g = Grid_2D.periodic(7, 7)
X = np.linspace(g.gx.xmin, g.gx.xmax, 100)
Y = np.linspace(g.gy.xmin, g.gy.xmax, 100)
X, Y = np.meshgrid(X, Y)
B = basis_fn(g.gx.dec.B, g.gy.dec.B)

fig = plt.figure()
ax = fig.gca(projection='3d')
for i, j, c in [(0, 0, 'r'), 
                (2, 2, 'g'), 
                (4, 4, 'b')]:
    Z = B[0, True](i, j)(X,Y)
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, color=c, alpha=0.4)

fig = plt.figure()
ax = fig.gca(projection='3d')
for i, j, c in [(0, 0, 'r'), 
                (2, 2, 'g'), 
                (4, 4, 'b')]:
    Z = B[0, False](i, j)(X,Y)
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, color=c, alpha=0.4)

fig = plt.figure()
ax = fig.gca(projection='3d')
for i, j, c in [(0, 0, 'r'), 
                (2, 2, 'g'), 
                (4, 4, 'b')]:
    Z = B[2, True](i, j)(X,Y)
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, color=c, alpha=0.4)

fig = plt.figure()
ax = fig.gca(projection='3d')
for i, j, c in [(0, 0, 'r'), 
                (2, 2, 'g'), 
                (4, 4, 'b')]:
    Z = B[2, False](i, j)(X,Y)
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, color=c, alpha=0.4)

plt.show()
