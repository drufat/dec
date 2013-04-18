from dec.spectral import *
from matplotlib.pyplot import *
from dec.plot import plot_bases_1d

N = 2
g = Grid_1D_Regular(N)
figure()
plot_bases_1d(g, g.xmin, g.xmax, "\kappa")

N = 5
g = Grid_1D_Regular(N)
figure()
plot_bases_1d(g, g.xmin, g.xmax, "\kappa")

figure()
plot_bases_1d(g, -g.xmax, g.xmax, "\kappa")

figure()
N = 9
g = Grid_1D_Regular(N)
plot_bases_1d(g, g.xmin, g.xmax, "\kappa")

show()