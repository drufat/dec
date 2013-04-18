from numpy import *
from matplotlib.pyplot import *

def plot_bases_1d(g, xmin, xmax, name):

	N = len(g.verts)	
	z = linspace(xmin, xmax, 100)
	B0, B1, B0d, B1d = g.basis_fn()
	
	subplot(221)
	title("${%s}^0_{%d,n}$" % (name,N))
	for i, b in enumerate(B0):
		plot(z, b(z), label=str(i))
		scatter(g.verts, 0*g.verts)
	
	subplot(222)
	title("${%s}^1_{%d,n}$" % (name,N))
	for i, b in enumerate(B1):
		plot(z, b(z)*g.delta[i], label=str(i))
		scatter(g.verts, 0*g.verts)
	
	subplot(223)
	title("$\widetilde{%s}^0_{%d,n}$" % (name,N))
	for i, b in enumerate(B0d):
		plot(z, b(z), label=str(i))
		scatter(g.verts, 0*g.verts)
		scatter(g.verts_dual, 0*g.verts_dual, color='red', marker='x')
	
	subplot(224)
	title("$\widetilde{{%s}}^1_{%d,n}$" % (name,N))
	for i, b in enumerate(B1d):
		plot(z, b(z)*g.delta_dual[i], label=str(i))
		scatter(g.verts, 0*g.verts)
		scatter(g.verts_dual, 0*g.verts_dual, color='red', marker='x')

if __name__ == '__main__':
	from dec.spectral import *
	from matplotlib.pyplot import *
	from dec.plot import plot_bases_1d
	N = 5
	g = Grid_1D_Periodic(N)
	plot_bases_1d(g, g.xmin, g.xmax, "\phi")
	show()
