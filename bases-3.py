from dec.spectral import *
from matplotlib.pyplot import *
from dec.plot import plot_bases_1d
N = 5
g = Grid_1D_Chebyshev(N)
plot_bases_1d(g, g.xmin, g.xmax, "\kappa")
show()