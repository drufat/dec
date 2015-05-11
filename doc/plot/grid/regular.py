import matplotlib.pyplot as plt
from dec.grid1 import *
import dec.plot
 
ax = plt.subplot(311)
g = Grid_1D.regular(3)
dec.plot.grid_1d(ax, g)
 
ax = plt.subplot(312)
g = Grid_1D.regular(5)
dec.plot.grid_1d(ax, g)
 
ax = plt.subplot(313)
g = Grid_1D.regular(15)
dec.plot.grid_1d(ax, g)
 
 
plt.show()
