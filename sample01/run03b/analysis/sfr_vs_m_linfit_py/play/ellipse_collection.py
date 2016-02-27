# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 23:27:20 2015

@author: Peter
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection

x = np.arange(10)
y = np.arange(15)
X, Y = np.meshgrid(x, y)

homey = np.hstack((X.ravel()[:,np.newaxis], Y.ravel()[:,np.newaxis]))

ww = X/10.0
ww[:] = 1.0
hh = Y/15.0
hh[:] = 0.333
aa = X*9
#aa[:] = -50.0

fig, ax = plt.subplots()

myalpha = ww
myalpha[:] = 0.5
ec = EllipseCollection(
                        ww,
                        hh,
                        aa,
                        units='x',
                        alpha = 0.1, 
                        offsets=homey,
                        transOffset=ax.transData)
simon = (X+Y).ravel()
ec.set_array(simon)
ax.add_collection(ec)
ax.autoscale_view()
ax.set_xlabel('X')
ax.set_ylabel('y')
cbar = plt.colorbar(ec)
cbar.set_label('X+Y')
plt.show()
