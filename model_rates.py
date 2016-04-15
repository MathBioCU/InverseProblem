# -*- coding: utf-8 -*-
#Created on Feb 18, 2016
#@author: Inom Mirzaev

"""
    Model rates and parameters used for generation of existence and stability maps
    of the population balance equations (PBEs) (see Mirzaev, I., & Bortz, D. M. (2015). 
    arXiv:1507.07127 ). 
"""


from __future__ import division
from scipy.integrate import quad, odeint
from scipy import interpolate
 
from functools import partial

import scipy.linalg as lin
import numpy as np


"""
    Number of CPUs used for computation of existence and stability regions.
    For faster computation number of CPUs should be set to the number of cores available on
    your machine.
"""

ncpus = 2

# Minimum and maximum floc sizes
x0 = 0
x1 = 1

tfinal = 10


#Ininitial guess for gamma function. Uniform distribution

def init_gam( y , x  ):
    
    out = 1 / x
    out[y>x] = 0
    
    return out    


# Post-fragmentation density distribution
def gam( y , x ):
    
    out = 6*y * ( x - y )  / (x**3)
    
    if type(x) == np.ndarray or type(y) == np.ndarray:        
        out[y>x] = 0
    #Should return a vector
    return out 
    

#Aggregation rate
def aggregation( x , y ):
    
    out = ( x ** ( 1/3 ) + y ** ( 1/3 ) ) **3 / (10**6)    
    #Should return a vector
    return out


    
#Removal rate    
def rem( x ):
     #Should return a vector
     return 1e-2*x**(1/3)

     
#Fragmentation rate
def fragm( x ):
    #Should return a vector
    return  1e-1 * x**(1/3)


#Initial condition    
def incond(x):
    
    return 1e4 * np.exp(x)



#Projection function for the initial condition
def ICproj( N ):
    
     dx=( x1 - x0 ) / N
     nu=x0 + np.arange(N+1) * dx
     
     out=np.zeros(N)
     
     for jj in range(N):
         out[jj]= quad( incond , nu[jj] , nu[jj+1] ) [0] / dx

     return out           


    
#Initializes uniform partition of (x0, x1) and approximate operator F_n
def initialization( N ):
    
    #delta x
    dx = ( x1 - x0 ) / N
    
    #Uniform partition into smaller frames
    nu = x0 + np.arange(N+1) * dx
    
    #Aggregation in
    Ain = np.zeros( ( N , N ) )
    
    #Aggregation out
    Aout = np.zeros( ( N , N ) )
    
    #Fragmentation in
    Fin = np.zeros( ( N , N ) )
    
    #Fragmentation out
    Fout = np.zeros( N )



    #Initialize matrices Ain, Aout,  Fin and Fout
    for mm in range( N ):
    
        for nn in range( N ):
            
            if mm>nn:
            
                Ain[mm,nn] = 0.5 * dx * aggregation( nu[mm] , nu[nn+1] )
            
            if mm + nn < N-1 :
                
                Aout[mm, nn] = dx * aggregation( nu[mm+1] , nu[nn+1] )
                    
            if nn > mm :
            
                Fin[mm, nn] = dx * gam( nu[mm+1], nu[nn+1] ) * fragm( nu[nn+1] )


    #Initialize matrix Fout
    Fout = 0.5 * fragm( nu[range( 1 , N + 1 ) ] ) + rem( nu[range( 1 , N + 1 )] )


    return ( Ain , Aout , Fin, Fout ,  nu , N , dx)





def odeRHS(y , t , P , N ,  Ain, Aout, Fout, nu , dx ):
    
    """Approximate operator for the right hand side of the evolution equation"""
   
    Fin = dx * np.triu(P.T , 1) * fragm( nu[range( 1 , N+1 ) ] )
    
    a = np.zeros_like(y)

    a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ]    


    out = np.sum( Ain * y * lin.toeplitz( np.zeros_like(y) , a).T + 
                  Fin * y - (Aout.T*y).T * y, axis = 1 ) - Fout * y   
    return out

  
def dataRHS(y , t , N , Ain , Aout , Fin , Fout ):
   
   
    a = np.zeros_like(y)

    a[range(1,len(a))] = y[range(len(y) - 1)]        
    
    return np.sum( Ain * lin.toeplitz( np.zeros_like(y) , a).T * y + Fin * y, axis = 1 ) - \
           np.dot( (Aout.T*y).T , y )- Fout * y     
   



Ain, Aout, Fin, Fout, nu, N, dx = initialization( 1000 )
mytime = np.linspace( 0 , tfinal , 1000 )

y0 = ICproj(N)

data_generator = partial( dataRHS , N=N , Ain=Ain , Aout=Aout , Fin=Fin , Fout=Fout )           
mydata = odeint( data_generator , y0 , mytime )

myfunc = interpolate.interp2d( nu[:-1] , mytime , mydata )

