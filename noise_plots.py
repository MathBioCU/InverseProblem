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

fnames = []

for file in os.listdir("data_files"):
    if file.endswith("noise.pkl"):
        fnames.append(file)


myfile = fnames[-1]

ext = '_alpha_'+str( myfile[-11] )

pkl_file = open(os.path.join( 'data_files' , myfile ) , 'rb')

data = cPickle.load( pkl_file )        
pkl_file.close()

a = []
# Total variance norm. See Gibbs and Su (2009) for the definition.
tv_norm = np.zeros( len(data) )
var_norm = np.zeros( len(data) )

sup_norm = np.zeros( len(data) )
max_norm = np.zeros( len(data) )


N = len(data[0][-1])
dx = ( mr.x1 - mr.x0 ) / N

for cnum in range(len(data)):
    
    a.append( data[cnum][0] )    
    tv_norm[cnum] = np.sum ( np.abs( data[cnum][-2]-data[cnum][-1] ) ) * (dx**2) 
    var_norm[cnum] = np.max (  np.sum(  np.abs( data[cnum][-2]-data[cnum][-1] ) , axis=1 ) ) * dx
    sup_norm[cnum] =  np.sum( np.max (  np.abs ( data[cnum][-2]-data[cnum][-1] ) , axis=1 ) ) *dx
    max_norm[cnum] =  np.max (  np.abs ( data[cnum][-2]-data[cnum][-1] ) )
    
a = np.asarray( a )
tv_norm = sup_norm

min_ind = np.argmin( tv_norm ) 

f_fit   = data[ min_ind ][-1]
f_true  = data[ min_ind ][-2]
f_init  = data[ min_ind ][-3]


plt.close('all')
my_cmap = plt.get_cmap('Set2')

fig1 = plt.figure(1)

plt.title('$F_{30} (x,\ y)$', fontsize=20, y=1.04)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$y$', fontsize=20)

plt.imshow(np.flipud(f_fit), interpolation='nearest', cmap=my_cmap , \
               vmin=0.001, vmax = np.max( ( np.max(f_fit) , np.max(f_true)  ) ) , extent=(0,1,0,1))
               

cbar_ax = fig1.add_axes([0.85, 0.13, 0.03, 0.75])
plt.colorbar(cax=cbar_ax)


fig2 = plt.figure(2)

plt.title('$F_{0} (x,\ y)$', fontsize=20, y=1.04)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$y$', fontsize=20)

plt.imshow( np.flipud(f_true), interpolation='nearest', cmap=my_cmap , \
               vmin=0.001, vmax = np.max( ( np.max(f_fit) , np.max(f_true)  ) ) , extent=(0,1,0,1))
               

cbar_ax = fig2.add_axes([0.85, 0.13, 0.03, 0.75])
plt.colorbar(cax=cbar_ax)

fig3 = plt.figure(3)

plt.title('Absolute error for $F_{0}$ and $F_{30}$', fontsize=16, y=1.04)


aa =  np.abs( f_fit - f_true  )

imgplot = plt.imshow(np.flipud( aa )  , interpolation='nearest', cmap='Reds' , \
               vmin = 0, vmax = np.max( aa )  , extent=(0,1,0,1))


cbar_ax = fig3.add_axes([0.85, 0.13, 0.03, 0.75])
plt.colorbar(cax=cbar_ax)      

plt.xlabel('$x$', fontsize=20)
plt.ylabel('$y$', fontsize=20)



plt.figure(4)

plt.plot(a , tv_norm ,  color='blue', linewidth=1 , marker='o' , markersize=5 )
x1, x2, y1, y2 = plt.axis()
plt.axis( (x1-1, x2+1 , y1, y2))
#plt.xticks( range(5, 31, 5) )

plt.ylabel( r'$\rho_{TV} \left ( F_{0} , \ F_{30} \right )$', fontsize=20 ) 
plt.xlabel( '$\sigma$', fontsize=20 )

fig_name = 'noise_error'+ext+'.png'
plt.savefig( os.path.join( 'images' , fig_name ) , dpi=400 , bbox_inches='tight' ) 


f, ax = plt.subplots(2, sharex=True)


grid = np.linspace( mr.x0 , mr.x1 , len(f_true) )

mm = int( len(f_true)/2  )

ax[0].plot(  grid, f_true[ mm ] , color='b' , linewidth=2 , label='True' )
ax[0].plot(  grid, f_fit[ mm ] , color='r'  , linewidth=2 , label='Fit')
ax[0].set_ylim( [ 0 , 1.1 ] )
ax[0].legend( loc = 'best' )
ax[0].set_ylabel( '$F(x,\  0.5)$' , fontsize=20)


mm = -1 

ax[1].plot(  grid, f_true[ mm ] , color='b' , linewidth=2 , label='True' )
ax[1].plot(  grid, f_fit[ mm ] , color='r'  , linewidth=2 , label='Fit')
ax[1].set_ylim([0,1.1])
ax[1].set_xlabel('$x$', fontsize=20)
ax[1].set_ylabel('$F(x,\  1.0)$', fontsize=20)

