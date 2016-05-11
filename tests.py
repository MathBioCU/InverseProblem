
from __future__ import division
from scipy import interpolate
from functools import partial
from scipy.integrate import odeint, cumtrapz, quad

import numpy as np 
import model_rates as mr

import matplotlib.pyplot as plt

import time


a = np.linspace( mr.x0 , mr.x1 , 10 ,  endpoint=False )
b = np.linspace( 0, mr.tfinal , 1000 )

newdata  = mr.myfunc( a , b )


Ain, Aout, Fin, Fout, nu, N, dx = mr.initialization( 30 )
mytime = np.linspace( 0 , mr.tfinal , 50 )
y0 = mr.ICproj( N )

start  = time.time()

data_generator = partial( mr.dataRHS , N=N , Ain=Ain , Aout=Aout , Fin=Fin , Fout=Fout )           
mydata = odeint( data_generator , y0 , mytime , full_output=True)


import odespy

#solver = odespy.Vode( data_generator , adams_or_bdf='bdf', order=1)
solver = odespy.BackwardEuler( data_generator)

solver.set_initial_condition( y0 )

u, t = solver.solve( mytime )

end = time.time()
print round( end -start , 3)


start  = time.time()

rk_gen = partial( mr.rk_dataRHS , N=N , Ain=Ain , Aout=Aout , Fin=Fin , Fout=Fout )    
rk_data = mr.rk_solver( rk_gen , y0 ,  mytime )

end = time.time()
print round( end -start , 3)

plt.close('all')
plt.figure(0)
dx1 = ( mr.x1 - mr.x0 ) / len(a)
plt.plot( np.sum(  dx1 * newdata , axis=1) , color='b')
plt.plot( np.sum(  dx * mydata[0] , axis=1) , color='r')

mu, sigma = 0, 1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)


Ain, Aout, Fin, Fout, nu, N, dx = mr.initialization( 30 )    

mytime = np.linspace( 0 , mr.tfinal , 50 )
y0 = mr.ICproj( N )

data_generator = partial( mr.dataRHS , N=N , Ain=Ain , Aout=Aout , Fin=Fin , Fout=Fout )           
yout = odeint( data_generator , y0 , mytime , full_output=True)[0]

sol = 0.5 * dx * ( yout[ : , :-1 ] + yout[:, 1:] )

mmm_data = mr.interp_data( nu, mytime)



xx , yy = np.meshgrid( nu[1:] , nu[1:] )
 
 # Initial guess of gamma function for the optimization   
init_P = mr.gam( xx , yy)    

  
    