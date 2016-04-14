
from __future__ import division
from scipy import interpolate
from functools import partial
from scipy.integrate import odeint

import numpy as np 
import model_rates as mr


x = np.arange(-5.01, 5.01, 0.1)
y = np.arange(-5.01, 5.01, 0.1)
xx, yy = np.meshgrid(x, y)
z = np.sin(xx**2+yy**2)

f = interpolate.interp2d(x, y, z, kind='cubic')

import matplotlib.pyplot as plt

xnew = np.arange(-5.01, 5.01, 0.2)
ynew = np.arange(-5.01, 5.01, 0.2)
znew = f( xnew , ynew )

plt.plot(x, z[0, :], 'ro-', xnew, znew[0, :], 'b-')

plt.show()



a = np.linspace( mr.x0 , mr.x1 , 10,  endpoint=False )
b = np.linspace( 0, mr.tfinal , 100 )

newdata  = mr. myfunc( a , b )