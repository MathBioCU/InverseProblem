# -*- coding: utf-8 -*-
"""
Created on May 10, 2016

@author: Inom Mirzaev

Simulates the forward problem and plots the results.
Uses the rates specified in model_rates.py file.
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import os
import model_rates as mr

from scipy.integrate import odeint
from functools import partial


#Extension to be attached to the output file
ext = '_alpha_'+str( mr.a )


plt.close('all')


#==============================================================================
# Simulations of the forwards problem
#==============================================================================

fig = plt.figure( 1 )

ax = fig.add_subplot(111)


num_plots = 5
a = int( len( mr.mytime) / num_plots )    
x = mr.nu[:-1]

linestyles = ['-', '--', '-.', ':']
lnst = 0

for nn in range(0, len(mr.mytime) , a)+[-1]:    

    y = mr.mydata[nn]

    mytxt = '$t=' + str( round(mr.mytime[nn] , 0) ) +'$'
    ax.plot( x ,  y , label=mytxt , linewidth=2  , linestyle = linestyles[ np.mod( lnst , 4) ] )
    lnst+=1

plt.legend(loc='best', fontsize=20)
plt.ylabel('$b(t,x) $', fontsize=25)
plt.xlabel('$x$', fontsize=25)

fig_name = 'simulation'+ext+'.png'
plt.savefig( os.path.join( 'images' , fig_name ) , dpi=400 , bbox_inches='tight' )


#==============================================================================
# Plot of the Gamma (pdf) and F (cdf) for fixed values of y 
#==============================================================================
f, ax = plt.subplots(2, sharex=True)

x = mr.nu[:-1]
yy = [0.2, 0.5, 1.0]


for mm in range( len(yy) ):
    
    y = mr.gam( x , yy[mm] )    
    mytxt = '$\Gamma(x,\ ' + str( yy[mm] ) +')$'
    ax[0].plot( x ,  y , label=mytxt , linewidth=2 , linestyle = linestyles[ mm ] )
    
    F_y  = np.cumsum( y )
    F_y = F_y / np.max( F_y )
    mytxt = '$F(x,\ ' + str( yy[mm] ) +')$'
    ax[1].plot( x ,  F_y , label=mytxt , linewidth = 2 , linestyle = linestyles[ mm ] )
    

ax[0].legend( loc='best' , fontsize=15 )
ax[0].set_ylabel( r'$\Gamma(x,\ y) $' , fontsize=20 )
myaxis = list( ax[0].axis() )
myaxis[-1]+=0.2
ax[0].axis( myaxis )

ax[1].legend( loc='lower right' , fontsize=15 )
ax[1].set_ylabel( r'$F(x,\ y) $' , fontsize=20 )
ax[1].set_xlabel( '$x$' , fontsize=20 )
myaxis = list( ax[1].axis() )
myaxis[-1]+=0.1
ax[1].axis( myaxis )



fig_name = 'gamma'+ext+'.png'
plt.savefig( os.path.join( 'images' , fig_name ) , dpi=400 , bbox_inches='tight' )


dimens  = np.linspace(10, 100, 20)
conv_error = np.array([])  

for ndim in range(len(dimens)):

    Ain, Aout, Fin, Fout, nu, N, dx = mr.initialization( int( dimens[ndim]) )
    mytime = np.linspace( 0 , 1 , 1000 )
    
    y0 = mr.ICproj(N)
    
    data_generator = partial( mr.dataRHS , N=N , Ain=Ain , Aout=Aout , Fin=Fin , Fout=Fout )           
    mydata = odeint( data_generator , y0 , mytime ,  rtol=1e-6, atol=1e-6 )
    #Approximate steady state calculated as a root of G_n           
    appr_sol = mydata
    
    grid_x = np.linspace( mr.x0 , mr.x1 , N )
    actual_sol =  mr.interp_func( grid_x , mytime )

    #L1 error between exact and approximate solutions
    L1_error =  np.trapz( np.trapz( np.abs( actual_sol - appr_sol )  , dx=dx, axis=1 ) , dx=mytime[1] )    
    conv_error = np.append( conv_error, L1_error ) 
     

#==============================================================================
# Generates error plot for the forward simulation.
#==============================================================================

fig = plt.figure(3)
ax = fig.add_subplot(111)

logx = np.log(  1/dimens )
logy = np.log( conv_error )

coeffs = np.polyfit( logx , logy ,deg=1 )

ax.loglog( 1/dimens ,  conv_error , linewidth=1, color='blue', marker='o', markersize=10 )

x1, x2, y1, y2 = plt.axis()
ax.axis([x1 - 0.1*x1, x2+0.01*x2, y1, y2])
ax.set_ylabel( '$\Vert u_{1000} - u_N  \Vert_{1}$', fontsize=25)
ax.set_xlabel( r'$\Delta x=\frac{ \overline{x} } {N}$', fontsize=25 )
mytext = 'Slope =$'+str( round(coeffs[0] , 1) ) + '$'

ax.text(0.7, 0.9,  mytext , style='italic',
        bbox={'facecolor':'#87CEFA', 'alpha':0.5, 'pad':10} , 
        transform=ax.transAxes, fontsize = 20)
  

fig_name = 'frwd_converg'+ext+'.png'
plt.savefig( os.path.join( 'images' , fig_name ) , dpi=400 , bbox_inches='tight' )





