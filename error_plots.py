# -*- coding: utf-8 -*-
"""
Created on May 15 2016

@author: Inom Mirzaev

Plots the generated data from parallel minimize.

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import cPickle, os
import model_rates as mr

fnames = []


#read all the files in the folder
for file in os.listdir("data_files"):
    if file.endswith("cobyla.pkl"):
        fnames.append(file)


#Open the specific file
myfile = fnames[ -1]
ext = '_alpha_' + str( myfile[-12] )
pkl_file = open(os.path.join( 'data_files' , myfile ) , 'rb')

data = cPickle.load( pkl_file )        
pkl_file.close()

a = []


# Total variance norm. See Gibbs and Su (2009) for the definition.
tv_norm = np.zeros( len(data) )


N = len(data[0][-1])
dx = ( mr.x1 - mr.x0 ) / N

for cnum in range(len(data)):
    
    a.append( data[cnum][0] )    
    #Compute total variation norm
    tv_norm[cnum] =  np.sum( np.max (  np.abs ( data[cnum][-2]-data[cnum][-1] ) , axis=1 ) ) *dx
    
a = np.asarray( a )

min_ind = 10

f_fit   = data[ min_ind ][-1]
f_true  = data[ min_ind ][-2]
f_init  = data[ min_ind ][-3]


plt.close('all')


#Colormap for the imshow plots
my_cmap = plt.get_cmap('Set2')


#==============================================================================
# plot of approximate conditional measure (cdf) 
#==============================================================================


fig1 = plt.figure(1)

plt.title('$F_{30} (x,\ y)$', fontsize=20, y=1.04)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$y$', fontsize=20)

plt.imshow(np.flipud(f_fit), interpolation='nearest', cmap=my_cmap , \
               vmin=0.001, vmax = np.max( ( np.max(f_fit) , np.max(f_true)  ) ) , extent=(0,1,0,1))
               

cbar_ax = fig1.add_axes([0.85, 0.13, 0.03, 0.75])
plt.colorbar(cax=cbar_ax)


fig_name = 'f_fit'+ext+'.png'

plt.savefig( os.path.join( 'images' , fig_name ) , dpi=400 , bbox_inches='tight' ) 


#==============================================================================
# plot of true conditional measure (cdf)
#==============================================================================

fig2 = plt.figure(2)

plt.title('$F_{0} (x,\ y)$', fontsize=20, y=1.04)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$y$', fontsize=20)

plt.imshow(np.flipud(f_true), interpolation='nearest', cmap=my_cmap , \
               vmin=0.001, vmax = np.max( ( np.max(f_fit) , np.max(f_true)  ) ) , extent=(0,1,0,1))
               

cbar_ax = fig2.add_axes([0.85, 0.13, 0.03, 0.75])
plt.colorbar(cax=cbar_ax)

fig_name = 'f_true'+ext+'.png'
plt.savefig( os.path.join( 'images' , fig_name ) , dpi=400 , bbox_inches='tight' ) 




#==============================================================================
#  Error between true and fit
#==============================================================================

fig3 = plt.figure(3)

plt.title('Absolute error for $F_{0}$ and $F_{30}$', fontsize=16, y=1.04)


aa =  np.abs( f_fit - f_true  )

imgplot = plt.imshow(np.flipud( aa )  , interpolation='nearest', cmap='Reds' , \
               vmin = 0, vmax = np.max( aa )  , extent=(0,1,0,1))


cbar_ax = fig3.add_axes([0.85, 0.13, 0.03, 0.75])
plt.colorbar(cax=cbar_ax)      

plt.xlabel('$x$', fontsize=20)
plt.ylabel('$y$', fontsize=20)


fig_name = 'relative_error'+ext+'.png'
plt.savefig( os.path.join( 'images' , fig_name ) , dpi=400 , bbox_inches='tight' ) 




#==============================================================================
# Error with respect to t_f
#==============================================================================
plt.figure(4)

plt.plot(a, tv_norm ,  color='blue', linewidth=1 , marker='o' , markersize=5 )

plt.ylabel( r'$\rho_{TV} \left ( F_{0} , \ F_{30} \right )$', fontsize=20 ) 
plt.xlabel( '$t_f$', fontsize=20 )

fig_name = 'tv_error'+ext+'.png'
plt.savefig( os.path.join( 'images' , fig_name ) , dpi=400 , bbox_inches='tight' ) 


#==============================================================================
# plot of cdf for fixed y
#==============================================================================
grid = np.linspace( mr.x0 , mr.x1 , len(f_true) )


f, ax = plt.subplots(2, sharex=True)

mm = int( len(f_true)/2  )

ax[0].plot(  grid, f_true[ mm ] , color='b' , linewidth=2 , label='True' )
ax[0].plot(  grid, f_fit[ mm ] , color='r'  , linewidth=2 , linestyle = '--' , label='Fit')
ax[0].set_ylim([0,1.1])
ax[0].legend(loc='best')
ax[0].set_ylabel('$F(x,\  0.5)$', fontsize=20)


mm = -1 

ax[1].plot(  grid, f_true[ mm ] , color='b' , linewidth=2 , label='True' )
ax[1].plot(  grid, f_fit[ mm ] , color='r'  , linewidth=2 , linestyle = '--' , label='Fit')
ax[1].set_ylim([0,1.1])
ax[1].set_xlabel('$x$', fontsize=20)
ax[1].set_ylabel('$F(x,\  1.0)$', fontsize=20)

fig_name = 'approximate_F'+ext+'.png'
plt.savefig( os.path.join( 'images' , fig_name ) , dpi=400 , bbox_inches='tight' ) 

