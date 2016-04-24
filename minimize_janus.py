# -*- coding: utf-8 -*-

"""
Created on Mon Aug 03 13:18:38 2015

@author: Inom Mirzaev
"""



from __future__ import division

import numpy as np
import time
import pickle
import model_rates as mr

from functools import partial
from multiprocessing import Pool
from scipy.integrate import odeint
from scipy.optimize import fmin_cobyla , minimize



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
   
    seed = init_P[ np.tril_indices(N) ]


    def P2Gamma(P):
        
        Gamma = np.zeros( ( N , N ) )
        Gamma[ np.tril_indices(N) ] = P
        
        return 1 - dx*np.sum( Gamma , axis=1 )

    def neg_ineqs(j):
        
        return lambda P : P2Gamma(P)[j] 
        
    def pos_ineqs(j):
        
        return lambda P : -P2Gamma(P)[j] 
        
    def ineqs(j):
        
        return lambda P: P[j]
    
 
    
    ineq1 = [ neg_ineqs(j) for j in range(N ) ] 
    ineq2 = [ pos_ineqs(j) for j in range(N)  ]
    ineq3 = [ineqs(j) for j in range( len(seed) ) ]
    
    cons = ineq1 + ineq2 + ineq3
    

    res = fmin_cobyla( optim_func , seed, cons  , maxfun=10**4 , rhobeg=1 , rhoend=1)     


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
    
    f_true[np.triu_indices(N) ]  = 1
    f_fit[np.triu_indices(N) ]  = 1
    
    gamma_error  = np.max ( np.abs( f_fit - f_true ) ) 
    
    return (nana, data_error,  res , gamma_error, a , f_init, f_true, f_fit)


result = estimator( 20 )

end = time.time()

print 'Total time elapsed   ' + str( round( (end - start) / 60, 2 ) )  + ' minutes'



