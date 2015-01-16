from dec.grid1 import *
from matplotlib.pyplot import *
from dec.plot import plot_bases_1d

N = 4
#g = Grid_1D_Periodic(N)
g = Grid_1D_Regular(N)
#g = Grid_1D_Chebyshev(N)

z = linspace(g.xmin, g.xmax, 100) #+ 1e-16
B0, B1, B0d, B1d = g.basis_fn()
H0, H1, H0d, H1d = hodge_star_matrix(g.projection(), g.basis_fn())
H1d = linalg.inv(H0)

#polynomial fit
#def poly_coeff(basis):
#    A = array([polyfit(z, b(z), len(basis)-1)[::-1] for i, b in enumerate(basis)])
#    return A
#print poly_coeff(g.B0)
#print poly_coeff(g.B1d)

figure()

A = linalg.inv(H0).T
U = array([b(z) for b in B1d])
V = dot(A, array([b(z) for b in B0]))

for u, v in zip(U, V):
    plot(z, u)
    plot(z, v, color='k')
    scatter(g.verts, 0*g.verts)
    scatter(g.verts_dual, 0*g.verts_dual, color='r', marker='x')

figure()

A = linalg.inv(H1).T
U = array([b(z) for b in B0d])
V = dot(A, array([b(z) for b in B1]))

for u, v in zip(U, V):
    plot(z, u)
    plot(z, v, color='k')
    scatter(g.verts, 0*g.verts)
    scatter(g.verts_dual, 0*g.verts_dual, color='r', marker='x')

show()
