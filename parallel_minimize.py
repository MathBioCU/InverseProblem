# -*- coding: utf-8 -*-

"""
Created on Mon Aug 03 13:18:38 2015

@author: Inom Mirzaev
"""



from __future__ import division

import numpy as np
import time
import cPickle, os
import model_rates as mr

from functools import partial
from multiprocessing import Pool
from scipy.integrate import odeint
from scipy.optimize import   fmin_cobyla




start = time.time()

print 'Start time', time.strftime( "%H:%M" , time.localtime() )

#Initialization of the matrices required for the simulations
def estimator( nana ):
    #Definition of the left part of the system of ODE
    Ain, Aout, Fin, Fout, nu, N, dx = mr.initialization( nana )    

    xx , yy = np.meshgrid( nu[1:] , nu[1:] )
 
     # Initial guess of gamma function for the optimization   
    init_P = mr.init_gam( xx , yy)    
    
    mytime = np.linspace(0, mr.tfinal , 50 + nana)

    y0 = mr.ICproj( N )
               
    data = mr.interp_data( nu, mytime)


    # Same as data_generator, except for the Fin matrix
    myderiv = partial( mr.odeRHS , Ain=Ain, Aout=Aout, Fout=Fout, nu = nu, dx = dx)
  
    def optim_func( P, data=data, y0=y0, N=N , t=mytime):
        
        Gamma = np.zeros( ( N , N ) )
        Gamma[ np.tril_indices(N, -1) ] = P
        
        yout = odeint( myderiv , y0 , t , args=( Gamma , N ) , 
              printmessg=False, rtol=1e-3, atol=1e-6 , full_output = False)

        """
        solver = odespy.BackwardEuler( myderiv , f_args=[ Gamma , N ] )
        solver.set_initial_condition( y0 )
        yout = solver.solve( t )[0] """
        
        fit =np.zeros_like( yout )
        fit[: , 0] = dx* yout[:, 0]        
        fit[: , 1:] = 0.5 * dx * ( yout[ : , :-1 ] + yout[:, 1:] )

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

                  
    yout = odeint( myderiv , y0 , mytime , args = ( G_fit , N ) )          
    
    data_error =  np.linalg.norm( yout - data , np.inf ) / np.linalg.norm( data , np.inf )    

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

    
    return (nana, data_error,  res , f_init, f_true, f_fit)

"""
if __name__ == '__main__':
    
    pool = Pool( processes = 2 )
    ey_nana = range( 5 , 20 )

    result = pool.map(estimator, ey_nana)
    
    fname = 'data_' + time.strftime( "%m_%d_%H_%M" , time.localtime() ) + '_cobyla.pkl'  
    output_file = open( os.path.join( 'data_files' , fname ) , 'wb')
    cPickle.dump(result, output_file)
    output_file.close()
"""

result = []

for nn in range(5, 31):
    result.append( estimator(nn) )
    print nn

fname = 'data_' + time.strftime( "%m_%d_%H_%M_" , time.localtime() ) + str(mr.a) + '_cobyla.pkl'  
output_file = open( os.path.join( 'data_files' , fname ) , 'wb')
cPickle.dump(result, output_file)
output_file.close()
end = time.time()
print 'Total time elapsed   ' + str( round( (end - start) / 60, 2 ) )  + ' minutes'





