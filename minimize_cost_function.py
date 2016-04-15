# -*- coding: utf-8 -*-

"""
Created on Mon Aug 03 13:18:38 2015

@author: Inom Mirzaev
"""



from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import model_rates as mr

from functools import partial
from multiprocessing import Pool
from matplotlib import gridspec
from scipy.integrate import odeint
from scipy.optimize import fmin_slsqp , fmin_cobyla



start = time.time()


#Initialization of the matrices required for the simulations

def estimator( nana ):
    #Definition of the left part of the system of ODE
    Ain, Aout, Fin, Fout, nu, N, dx = mr.initialization( nana )    

    xx , yy = np.meshgrid( nu[1:] , nu[1:] )
 
 
     # Initial guess of gamma function for the optimization   
    init_P = mr.init_gam( xx , yy)    
    
    mytime = np.linspace(0, mr.tfinal , 100 + nana)

    y0 = mr.ICproj( N )
               
    data = mr.myfunc( nu[:-1] , mytime )


    # Same as data_generator, except for the Fin matrix
    myderiv = partial( mr.odeRHS , Ain=Ain, Aout=Aout, Fout=Fout, nu = nu, dx = dx)
    
    def optim_func( P, data=data, y0=y0, N=N , t=mytime):
        
        Gamma = np.zeros( ( N , N ) )
        Gamma[ np.tril_indices(N) ] = P
    
        yout = odeint( myderiv , y0 , t , args=( Gamma , N ) , printmessg=False)

        return  np.sum( ( yout - data ) **2 ) 

    # Inequality constraints. In particlular, p_lm >=0 at each iteration

    def funineq(P, N=N):
        Gamma = np.zeros( ( N , N ) )
        Gamma[ np.tril_indices(N) ] = P
        return np.concatenate ( ( 1 - dx*np.sum( Gamma , axis=1 ) , dx*np.sum( Gamma , axis=1 ) -1 ,  P ) )
    
    
    # Equation constraints, this part makes sure that upper triangular part is zero. 
    # Moreover, it guarantees gamma is a probability function    

    def funeqcons(P, N=N, dx=dx):
        
        Gamma = np.zeros( ( N , N ) )
        Gamma[ np.tril_indices(N) ] = P
        
        return ( dx*np.sum( Gamma , axis=1 ) -1 )**2
    
    seed = init_P[ np.tril_indices(N) ]
    
    """
    res = fmin_slsqp( optim_func , seed , f_eqcons = funeqcons, 
                  #f_ieqcons =   funineq ,    
                  full_output = 0 , epsilon = 1e-7 , iter=1000 , bounds = len(seed)*[(-1 , np.inf ) ]   )
    """

                              
    res = fmin_cobyla( optim_func , seed, funineq , maxfun=10**5 , catol=0.01, rhobeg=0.1 , rhoend=0.01)                  
    a = np.zeros( ( N , N ) )
    a[ np.tril_indices(N) ] = res
                  
    yout = odeint( myderiv , y0 , mytime , args = ( a , N ) )          
    
    data_error =  np.linalg.norm( yout - data , np.inf ) / np.linalg.norm( data , np.inf )    

    f_fit = np.zeros( ( N , N ) )
    f_fit  = dx * a
    
    f_true = np.zeros( ( N , N ) )    
    true_P = mr.gam( xx , yy )
    f_true  = dx * true_P
    
    f_init = np.zeros( ( N , N ) ) 
    f_init = dx * init_P

    for col in range(1,  nana ):
        
        f_fit[:, col ] = f_fit[:, col - 1 ] + dx * a[:, col]
        f_init[:, col ] = f_init[:, col - 1 ] + dx * init_P[:, col]
        f_true[:, col ] = f_true[:, col - 1 ] + dx * true_P[:, col]
    
    gamma_error  = np.max ( np.abs( f_fit - f_true ) ) 
    
    return (nana, data_error,  gamma_error, a , f_init, f_true, f_fit)


result = estimator( 15 )

end = time.time()

print 'Total time elapsed   ' + str( round( (end - start) / 60, 2 ) )  + ' minutes'


plt.close('all')

fig=plt.figure( 0 , figsize=(12,6) )
gs=gridspec.GridSpec(1,2)

ax0=plt.subplot(gs[0,0])
ax1=plt.subplot(gs[0,1])

ax0.set_title('$\Gamma_{\mathrm{data}} (x,\ y)$', fontsize=25, y=1.04)
ax0.set_xlabel('$x$', fontsize=20)
ax0.set_ylabel('$y$', fontsize=20)

ax1.set_title('$\Gamma_{ \mathrm{fit} } (x,\ y)$', fontsize=25 , y=1.04)
ax1.set_xlabel('$x$', fontsize=20)
ax1.set_ylabel('$y$', fontsize=20)


f_true = result[-2]
f_fit  = result[-1]
f_init = result[-3]

ax0=ax0.imshow(np.flipud(f_true), interpolation='nearest', cmap=plt.cm.Set2 , \
               vmin=0, vmax = np.max( ( np.max( f_fit ) , np.max( f_true)  ) ) , extent=(0,1,0,1))

ax1=ax1.imshow(np.flipud( f_fit ), interpolation='nearest', cmap=plt.cm.Set2 , \
               vmin = 0, vmax = np.max( ( np.max(f_fit ) , np.max( f_true )  ) ), extent=(0,1,0,1))


cbar_ax = fig.add_axes( [0.92, 0.15, 0.02, 0.7] )
fig.colorbar(ax0, cax=cbar_ax)

plt.figure( 1 )

mm = int( result[0] / 2  )

plt.plot(  f_true[ mm ] , color='b'  )
plt.plot(  f_fit[ mm ] , color='r'  )
plt.plot(  f_init[ mm ] , color='g'  )

plt.figure( 2 )

mm = -1 

plt.plot(  f_true[ mm ] , color='b'  )
plt.plot(  f_fit[ mm ] , color='r'  )
plt.plot(  f_init[ mm ] , color='g'  )

Gamma = result[-4]





