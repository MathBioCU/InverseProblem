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
from scipy.optimize import fmin_slsqp



start = time.time()


#Initialization of the matrices required for the simulations

def estimator( nana ):
    #Definition of the left part of the system of ODE
    Ain, Aout, Fin, Fout, nu, N, dx = mr.initialization( nana )    

    xx , yy = np.meshgrid( nu[1:] , nu[1:] )
 
 
     # Initial guess of gamma function for the optimization   
    init_P = mr.init_gam( xx , yy)    
    
    mytime = np.linspace(0, mr.tfinal , 40 + nana)

    y0 = mr.ICproj( N )
               
    data = mr.myfunc( nu[1:] , mytime )


    # Same as data_generator, except for the Fin matrix
    myderiv = partial( mr.odeRHS , Ain=Ain, Aout=Aout, Fout=Fout, nu = nu, dx = dx)
    
    def optim_func( P, data=data, y0=y0, N=N , t=mytime):
    
        P = np.reshape(P, (N, N))
    
        yout = odeint( myderiv , y0 , t , args=( P , N ) , printmessg=False)

        return  np.sum( ( yout - data ) * ( yout - data ) ) 

    # Inequality constraints. In particlular, p_lm >=0 at each iteration

    def funineq(P):
    
        return P


    # Equation constraints, this part makes sure that upper triangular part is zero. 
    # Moreover, it guarantees gamma is a probability function    

    def funeqcons(P, N=N, dx=dx):
    
        P = np.reshape( P, ( N , N ) )    
        out = np.sum( np.abs( np.triu( P , 1 ) ) ) 
        
        return np.append( dx * np.sum( P , axis=1 ) - np.ones( N )  , out )
        
    
    if nana>29:
        
        ppp = 1e-2
        
    else:
        
        ppp = 1e-3

    
    res = fmin_slsqp( optim_func , np.reshape( init_P , N*N ) , f_eqcons = funeqcons, \
                  f_ieqcons =   funineq ,  disp = False ,  
                  full_output = 1 , epsilon = ppp  )
                  
    yout = odeint( myderiv , y0 , mytime , args = ( np.reshape( res[0] , (N , N)  ) , N ) )          
    
    data_error =  np.linalg.norm( yout - data , np.inf ) / np.linalg.norm( data , np.inf )    


    a = np.reshape( res[0], ( N , N ) )
    a[a<0] = 0

    f_true = np.zeros((N, N))


    for mm in range(N):
    
        for nn in range(N):
        
            if nn >= mm:
            
                f_true[mm, nn] = 1
            
            else:
            
                f_true[mm, nn] = 2 * np.arctan( np.sqrt( nu[nn+1] ) / np.sqrt( nu[mm+1] - nu[nn+1] ) ) / np.pi
       
    f_fit  = dx * a 

    for col in range(1,  nana ):
        f_fit[:, col ] = f_fit[:, col - 1 ] + dx * a[:, col]
    
    
    gamma_error  = np.max ( np.abs( f_fit - f_true ) ) 
    
    return (nana, data_error, gamma_error, f_true, f_fit)


result = estimator( 30 )

end = time.time()

print 'Total time elapsed   ' + str( round( (end - start) / 60, 2 ) )  + ' minutes'


plt.close('all')

fig=plt.figure( 0 , figsize=(12,6))
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

ax0=ax0.imshow(np.flipud(f_true), interpolation='nearest', cmap=plt.cm.Set2 , \
               vmin=0, vmax = np.max( ( np.max( f_fit ) , np.max( f_true)  ) ) , extent=(0,1,0,1))

ax1=ax1.imshow(np.flipud( f_fit ), interpolation='nearest', cmap=plt.cm.Set2 , \
               vmin = 0, vmax = np.max( ( np.max(f_fit ) , np.max( f_true )  ) ), extent=(0,1,0,1))


cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(ax0, cax=cbar_ax)







