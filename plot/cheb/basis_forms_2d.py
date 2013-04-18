from mpl_toolkits.mplot3d import axes3d
from matplotlib.pyplot import *
from numpy import *
from dec.spectral2 import Grid_2D_Periodic

X = linspace(0, 2*pi, 100)
Y = linspace(0, 2*pi, 100)
X, Y = meshgrid(X, Y)
g = Grid_2D_Periodic(7, 7)

fig = figure()
ax = fig.gca(projection='3d')
for i, j, c in [(0, 0, 'r'), 
                (2, 2, 'g'), 
                (4, 4, 'b')]:
    Z = g.B0[i, j](X,Y)
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, color=c, alpha=0.4)

fig = figure()
ax = fig.gca(projection='3d')
for i, j, c in [(0, 0, 'r'), 
                (2, 2, 'g'), 
                (4, 4, 'b')]:
    Z = g.B0d[i, j](X,Y)
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, color=c, alpha=0.4)

fig = figure()
ax = fig.gca(projection='3d')
for i, j, c in [(0, 0, 'r'), 
                (2, 2, 'g'), 
                (4, 4, 'b')]:
    Z = g.B2[i, j](X,Y)
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, color=c, alpha=0.4)

fig = figure()
ax = fig.gca(projection='3d')
for i, j, c in [(0, 0, 'r'), 
                (2, 2, 'g'), 
                (4, 4, 'b')]:
    Z = g.B2d[i, j](X,Y)
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, color=c, alpha=0.4)

show()
