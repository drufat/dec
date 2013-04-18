from dec.spectral import *
from matplotlib.pyplot import *

N = 5
g = Grid_1D_Periodic(N)
R0, _, _, _ = reconstruction(g.basis_fn())

random.seed(seed=1)
y = random.rand(N)

x = linspace(0, 2*pi, 100)
plot(x, R0(y)(x)  , color='k')
scatter(g.verts, y, marker='o', color='k')

n = 2
g2 = Grid_1D_Periodic(N+2*n)
y2 = real( Finv(extend(F(y), n)) )
plot(x, R0(y)(x)+1, color='k')
scatter(g2.verts, y2+1, marker='o', color='k')

#n = 2
#g3 = Grid_1D_Periodic(N-2*n)
#y3 = real( Finv(unextend(F(y), n)) )
#scatter(g3.verts, y3, marker='x', color='g')

frame = gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)

show()