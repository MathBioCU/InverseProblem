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
import odespy

from scipy import interpolate
from functools import partial
from multiprocessing import Pool
from matplotlib import gridspec
from scipy.integrate import odeint
from scipy.optimize import   fmin_cobyla



start = time.time()

print 'Start time', time.strftime( "%H:%M" , time.localtime() )

#Initialization of the matrices required for the simulations

#==============================================================================
# Generate data
#==============================================================================

fine_N = 1000

fine_t = 10000

tfinal = 20

Ain, Aout, Fin, Fout, nu, N, dx = mr.initialization( fine_N )
mytime = np.linspace( 0 , tfinal , fine_t )

y0 = mr.ICproj( N )

data_generator = partial( mr.dataRHS , N=N , Ain=Ain , Aout=Aout , Fin=Fin , Fout=Fout )           
mydata = odeint( data_generator , y0 , mytime ,  rtol=1e-6, atol=1e-6 )

interp_x  = np.linspace( mr.x0 , mr.x1 , fine_N )
interp_func = interpolate.interp2d( interp_x , mytime , mydata )


def interp_data( nu , mytime , mu=0 , sigma=20 ):

    data = np.zeros( ( len(mytime) , len(nu) - 1 ) )
    
    for mm in range( len(nu) - 1):
        
        int_grid = np.linspace( nu[mm] , nu[mm+1] )
        
        data[ : , mm] = np.trapz( interp_func( int_grid , mytime ) , int_grid , axis=1 )
        
    #Add some normally distributed error
    if sigma>0:    
        noise = np.random.normal( mu , sigma , data.shape )
        data += noise
        
    return data


def estimator( tfinal ):
    
    #Definition of the left part of the system of ODE
    Ain, Aout, Fin, Fout, nu, N, dx = mr.initialization( 30 )    

    xx , yy = np.meshgrid( nu[1:] , nu[1:] )
 
    # Initial guess of gamma function for the optimization   
    init_P = mr.init_gam( xx , yy)    
    
    
    data_t = np.linspace(0, tfinal , 20)
    data_x = np.linspace( mr.x0 , mr.x1 , 11)
 
               
    data = interp_data( data_x , data_t)
    
    
    # Same as data_generator, except for the Fin matrix
    myderiv = partial( mr.odeRHS , Ain=Ain, Aout=Aout, Fout=Fout, nu = nu, dx = dx)

    y0 = mr.ICproj( N )
    mytime = np.linspace( 0 , tfinal , 100 ) 
  
    def optim_func( P, data=data, y0=y0, N=N , t=mytime ,  data_x = data_x , data_t = data_t):
        
        Gamma = np.zeros( ( N , N ) )
        Gamma[ np.tril_indices(N, -1) ] = P
        
        yout = odeint( myderiv , y0 , t , args=( Gamma , N ) , 
              printmessg=False, rtol=1e-3, atol=1e-5 , full_output = False)
        
        
        interp_x = np.linspace(mr.x0 , mr.x1 , N)
        func = interpolate.interp2d( interp_x , t , yout )


        fit = np.zeros_like( data )
    
        for mm in range( len(data_x) - 1):
            
            int_grid = np.linspace( data_x[mm] , data_x[mm+1] , 10)
            
            fit[ : , mm] = np.trapz( func( int_grid , data_t ) , int_grid , axis=1 )

        """
        solver = odespy.BackwardEuler( myderiv , f_args=[ Gamma , N ] )
        solver.set_initial_condition( y0 )
        yout = solver.solve( t )[0] """
        

        return  np.sum( ( fit - data ) **2 ) 

     
    seed = init_P[ np.tril_indices(N, -1) ]
    
    def P2Gamma(P):
        
        Gamma = np.zeros( ( N , N ) )
        Gamma[ np.tril_indices(N,-1) ] = P
        
        return 1 - dx*np.sum( Gamma , axis=1 )

    def neg_ineqs(j):
        
        return lambda P : P2Gamma(P)[j] 
        
    def ineqs(j):
        
        return lambda P: P[j]
     
    
    ineq1 = [ neg_ineqs(j) for j in range(N ) ] 
    ineq2 = [ineqs(j) for j in range( len(seed) ) ]    
    cons = ineq2 + ineq1
    
    res = fmin_cobyla( optim_func , seed, cons  , maxfun=10000 , rhobeg=1 , rhoend=1)     

    G_fit = np.zeros( ( N , N ) )
    G_fit[ np.tril_indices(N,-1) ] = res

    f_fit = np.zeros( ( N , N ) )
    f_fit  = np.cumsum( dx * G_fit, axis=1)
    f_fit[np.triu_indices(N) ]  = 1

    
    f_true = np.zeros( ( N , N ) )    
    true_P = mr.gam( xx , yy )
    f_true  =np.cumsum( dx * true_P , axis=1)
    f_true[np.triu_indices(N) ]  = 1
    
    f_init = np.zeros( ( N , N ) ) 
    f_init = np.cumsum( dx * init_P , axis=1)
    f_init[np.triu_indices(N) ]  = 1 

    cost_func = optim_func( res )
    
    return (tfinal,  res , cost_func ,  f_init, f_true, f_fit)


result = estimator( 10 )

end = time.time()

print 'Total time elapsed   ' + str( round( (end - start) / 60, 2 ) )  + ' minutes'


plt.close('all')

fig=plt.figure( 0 , figsize=( 12 , 6 ) )
gs=gridspec.GridSpec( 1 , 2 )

ax0=plt.subplot( gs[0,0] )
ax1=plt.subplot( gs[0,1] )

ax0.set_title('$\Gamma_{\mathrm{data}} (x,\ y)$', fontsize=25, y=1.04)
ax0.set_xlabel( '$x$', fontsize=20 )
ax0.set_ylabel( '$y$', fontsize=20 )

ax1.set_title( '$\Gamma_{ \mathrm{fit} } (x,\ y)$' , fontsize=25 , y=1.04)
ax1.set_xlabel( '$x$' , fontsize=20)
ax1.set_ylabel( '$y$' , fontsize=20)


f_true = result[-2]
f_fit  = result[-1]
f_init = result[-3]

ax0=ax0.imshow(np.flipud(f_true), interpolation='nearest', cmap=plt.cm.Set2 , \
               vmin=0, vmax = np.max( ( np.max( f_fit ) , np.max( f_true)  ) ) , extent=(0,1,0,1))

ax1=ax1.imshow(np.flipud( f_fit ), interpolation='nearest', cmap=plt.cm.Set2 , \
               vmin = 0, vmax = np.max( ( np.max(f_fit ) , np.max( f_true )  ) ), extent=(0,1,0,1))


cbar_ax = fig.add_axes( [ 0.92 , 0.15 , 0.02 , 0.7 ] )
fig.colorbar( ax0 , cax = cbar_ax )

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




