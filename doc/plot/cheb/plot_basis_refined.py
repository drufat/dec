from pylab import *
from dec.grid1 import *


#G = Grid_1D_Periodic; s = pi; b = phi1
G = Grid_1D_Chebyshev; s = 1; b = psi1

x = linspace(-1.5 * s, 1.5 * s, 1000)

n = 4

def plot_seperate(n, v, c='k'):
    g = G(n, -s, +s)
    scatter(g.verts, 0*g.verts, color=c)
    h = diff(g.verts)[0]
    for i in v:
        plot(x, b(n, i, x)*h, color=c)

def plot_together(n, v, c='r'):
    g = G(n, -s, +s)
    scatter(g.verts, 0*g.verts, color=c)
    h = diff(g.verts)[0]
    y = 0
    for i in v:
        y += b(n, i, x)*h
    plot(x, y, color=c)

plot_seperate(n, (1,),  c='k')
plot_seperate(2*n-1, (2,3), c='r')
plot_together(2*n-1, (2,3), c='b')

show()