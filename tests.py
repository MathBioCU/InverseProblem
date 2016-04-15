
from __future__ import division
from scipy import interpolate
from functools import partial
from scipy.integrate import odeint

import numpy as np 
import model_rates as mr

import matplotlib.pyplot as plt



a = np.linspace( mr.x0 , mr.x1 , 30 ,  endpoint=False )
b = np.linspace( 0, mr.tfinal , 100 )

newdata  = mr.myfunc( a , b )


Ain, Aout, Fin, Fout, nu, N, dx = mr.initialization( 30 )
mytime = np.linspace( 0 , mr.tfinal , 100 )
y0 = mr.ICproj( N )

data_generator = partial( mr.dataRHS , N=N , Ain=Ain , Aout=Aout , Fin=Fin , Fout=Fout )           
mydata = odeint( data_generator , y0 , mytime )

plt.close('all')
plt.figure(0)
dx1 = ( mr.x1 - mr.x0 ) / len(a)
plt.plot( np.sum(  dx1 * newdata , axis=1) , color='b')
plt.plot( np.sum(  dx * mydata , axis=1) , color='r')