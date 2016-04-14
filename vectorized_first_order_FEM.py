"""
Created on Mon June 15 13:29:46 2015

@author: Inom Mirzaev
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import quad
from scipy.integrate import odeint
from matplotlib import animation
from scipy.linalg import toeplitz
import time


N=100
xmin=0
xmax=1

dx=(xmax-xmin)/(N)
xu=xmin+dx*np.arange(N+1)

def q(x):
    
    return x/(10**6)
    
def g(x):
    
    xmin=1
    xmax=100    
    out = (x-xmin)*(xmax-x)/(10**4)
    
    return out


        
def gam( y , x , xmin = xmin):
    

    
    if x != xmin:
        out = 1 / (x-xmin)
    else:
        out = 0
        
    return out    
    

def rem(x , xmin = xmin):
    
    return (x-xmin)**(1/3) / (10**1)

def kf( x , xmin = xmin ):
    
    out = ( x - xmin ) ** (1/3) / (10**1)

    return out
    
def ka(x,y):


    
    out=(x**(1/3)+y**(1/3))*(x**(-1/3)+y**(-1/3))/(10**8)    
    
    return out

###########################################################################    
##                                                                       
##          Initial condition                                                     

     
def incond(x):
    
    out = x
    
    return out
 
start = time.time()   
###########################################################################  
##                                                                       
##          Initial condition initialization                                      

    
y0 = incond( xu )


###########################################################################
##                                                                       
##          Projection function for fragmentation  and aggregation                



Fin = np.zeros( (N+1 , N+1) )
Fout = np.zeros(N+1)

Fout[0] = quad(kf, xu[0], xu[1])[0] / dx
Fout[N] = quad( kf, xu[N-1], xu[N])[0] / dx

Gn=np.zeros((N+1,N+1))

Ain1 = np.zeros((N+1,N+1))
Ain2 = np.zeros((N+1,N+1))


Aout = np.zeros((N+1,N+1))


for mm in range(N+1):
    
    if mm < N:    
        
        Gn[mm,mm] = - g( (xu[mm] + xu[mm+1]) / 2 ) / dx
        Gn[mm+1,mm] = g( ( xu[mm] + xu[mm+1] ) / 2 ) / dx    
    
    if mm > 0 and mm < N:
        
        Fout[mm] = quad(kf, xu[mm-1], xu[mm+1])[0] / (2 * dx )

    for nn in range(N+1):
        
        if mm < nn and nn != N:
            
            Fin[mm,nn] = 0.5 * gam(xu[mm], ( xu[nn-1] + xu[nn] ) / 2 ) *kf( ( xu[nn-1] + xu[nn] ) / 2 ) + \
                         gam(xu[mm], xu[nn]) * kf( xu[nn] ) + \
                         0.5 * gam(xu[mm], ( xu[nn] + xu[nn+1] ) / 2 ) *kf( ( xu[nn] + xu[nn+1] ) / 2 )
            
        if mm == nn and nn != N:
            
                         0.5 * gam( xu[mm] , ( xu[nn] + xu[nn+1] ) / 2) * kf( ( xu[nn] + xu[nn+1] ) / 2 )
            
        if nn == N and mm != nn:
            
            Fin[mm,nn] = 0.5 * gam( xu[mm] , xu[nn-1] ) * kf( xu[nn-1] ) + \
                         0.5 * gam( xu[mm] , ( xu[nn-1] + xu[nn] ) / 2) * kf( ( xu[nn-1] + xu[nn] ) / 2 )
        
        if nn != 0 and mm + nn < N-1:       
            
            Aout[mm,nn] = 0.5 * ka( xu[mm] , ( xu[nn-1] + xu[nn]) / 2 ) + ka( xu[mm] , xu[nn] ) + \
                          0.5 * ka( xu[mm], ( xu[nn] + xu[nn+1] ) / 2) 
        
        if nn==0:
            
            Aout[mm,nn] = 0.5 * ( ka( xu[mm], xu[0] ) + ka( xu[mm] , ( xu[0] + xu[1] ) / 2) )
                                
        if mm + nn == N-1:
            
            Aout[mm,nn] = 0.5 * ( ka( xu[mm], xu[N-mm-1] ) + ka( xu[mm] , ( xu[N-mm-1] + xu[N-mm-2] ) / 2) )
            
        if mm - nn == 1:
            
            Ain1[mm, nn] = ka( xu[nn] , (xu[0] + xu[1] ) / 2 )
            
        if mm - nn > 1:
            
            Ain1[mm, nn] = ka( xu[nn] , ( xu[mm - nn -1] + xu[mm - nn] ) / 2 ) + \
                           ka( ( xu[nn] + xu[nn+1] ) / 2 , xu[mm - nn - 1] )
         
        if mm > 0  and nn==0:
            
            Ain2[mm, nn] = 0.5 * ka( xu[nn], ( xu[mm-1] + xu[mm] ) / 2 )
            Ain2[mm,mm] = Ain2[mm,nn]                  
        
        if nn != 0 and mm-nn>0:
            
            Ain2[mm, nn] = ka( ( xu[nn-1] + xu[nn] ) / 2, xu[mm-nn] ) + \
                           ka( xu[nn] , (xu[mm-nn-1] + xu[mm-nn] ) / 2 )
            
            
            
            

Fout = Fout / 2


Fin[N,N] = 0.5 * gam( xu[N] , xu[N] ) * kf( xu[N] ) + \
                         0.5 * gam( xu[N] , ( xu[N-1] + xu[N] ) / 2) * kf( ( xu[N-1] + xu[N] ) / 2 )
Fin = Fin * dx / 2




Gn[0,:] = 2 * Gn[0,:]
Gn[N,N] = g(xu[N])/dx
Gn[N,:] = 2* Gn[N,:]

#Ain[0 , 0] = ka( xu[0] , ( xu[0] + xu[1] ) / 2 ) 

Ain1 = Ain1 * dx / 8
Ain2 = Ain2 * dx / 8
 
Aout = Aout * dx / 2

#Fin = 0
#Fout = 0
#Gn = 0              
   
def deriv( y , t , Ain1, Ain2,  Fin, Fout, Gn):
    
    
    #out = np.dot( np.dot( Fin , np.diag(y) ) + \
    #      Ain * np.dot( toeplitz(np.zeros_like(y) , y).T, np.diag(y) ), np.ones_like(y) ) - \
    #      Fout*y + np.dot( Gn -  np.dot( np.diag(y) , Aout ) , y )
    
    a = np.zeros_like(y)
    a[range(1,len(a))] = y[range(len(y) - 1)]
          
    out = np.sum (Fin * y + \
                  Ain1 * (toeplitz(np.zeros_like(y), a).T) * y + \
                  Ain2 * (toeplitz(np.zeros_like(y), y).T) * y, axis=1 ) - \
          Fout*y + np.dot( Gn -  (Aout.T*y).T , y )
                            
    return  out
    
t=np.arange( 0 , 100 , 0.01 )

yout=odeint( deriv , y0 , t , args=(Ain1, Ain2, Fin , Fout, 0) )
 
end = time.time()

print end - start

    
 
 
#########################################################################
##          
##          Retrieving total mass info.                                          
##          np.dot(yout[t,:]*aa) gives total mass of the system at time t       


aa = np.zeros(N+1)
aa[0] = 2 * xu[0] + xu[1]
aa[N] = xu[N-1] + 2 * xu[N]
aa[range(1,N)] = xu[range(N-1)] +4 * xu[range(1,N)] + xu[range(2,N+1)]
aa = aa * dx / 6

##########################################################################
##
##          Retrieving total number of particles.                                
##          np.dot(yout[t,:],bb) gives total number of the particles             
##          in the system at time t                                              


bb = dx * np.ones(N+1)
bb[0] = bb[0] / 2
bb[N] = bb[N] / 2



plt.close()
fig = plt.figure(figsize=(12,12))
gs = gridspec.GridSpec(2,2)

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])



def plotfun(ax,x,y,ttl,xlab, ylab):
    ax.plot(x,y, linewidth='2')
    ax.set_title(ttl,fontsize=16)
    ax.set_xlabel(xlab, fontsize=16)
    ax.set_ylabel(ylab,fontsize=16)
    return ax

ax0 = plotfun(ax0, xu, xu*yout[0,:], 'Initial Condition','$x$', '$x\cdot b(t,x)$')
ax1 = plotfun(ax1, t, np.dot( yout, bb ),'Evolution of total number of particles','$t$', '$M_0(t)$')
ax2 = plotfun(ax2, xu, xu*yout[len(t)-1,:], 'Size distribution at the end','$x$', '$x \cdot b(t, x)$')
ax3 = plotfun(ax3, t, np.dot( yout , aa),'Evolution of total mass', '$t$', '$M_1(t)$')


    
plt.tight_layout()
rect = fig.patch
rect.set_facecolor('white')
plt.show()

