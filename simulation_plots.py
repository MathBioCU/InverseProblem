# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 17:04:00 2015

@author: Inom
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import cPickle, os
import model_rates as mr


ext = '_alpha_'+str( mr.a )


plt.close('all')
fig = plt.figure( 0 )

ax = fig.add_subplot(111)


num_plots = 5
a = int( len( mr.mytime) / num_plots )    

x = mr.nu[:-1]

for nn in range(0, len(mr.mytime) , a)+[-1]:    

    y = mr.mydata[nn]

    mytxt = '$t=' + str( round(mr.mytime[nn] , 0) ) +'$'
    ax.plot( x ,  y , label=mytxt , linewidth=1 )

plt.legend(loc='best', fontsize=15)
plt.ylabel('$b(t,x) $', fontsize=20)
plt.xlabel('$x$', fontsize=20)

fig_name = 'simulation'+ext+'.png'
plt.savefig( os.path.join( 'images' , fig_name ) , dpi=400 , bbox_inches='tight' )


f, ax = plt.subplots(2, sharex=True)

x = mr.nu[:-1]

for yy in [0.2, 0.5, 1.0]:
    
    y = mr.gam( x , yy)    
    mytxt = '$\Gamma(x,\ ' + str( yy ) +')$'
    ax[0].plot( x ,  y , label=mytxt , linewidth=2)
    
    F_y  = np.cumsum( y )
    F_y = F_y / np.max( F_y )
    mytxt = '$F(x,\ ' + str( yy ) +')$'
    ax[1].plot( x ,  F_y , label=mytxt , linewidth=2)
    

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




