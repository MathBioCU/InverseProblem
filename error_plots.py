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
    if file.endswith("cobyla.pkl"):
        fnames.append(file)


myfile = fnames[ -1]

ext = '_alpha_' + str( myfile[-12] )

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

#min_ind = np.argmin( tv_norm ) 
min_ind = 10

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


fig_name = 'f_fit'+ext+'.png'

plt.savefig( os.path.join( 'images' , fig_name ) , dpi=400 , bbox_inches='tight' ) 

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





plt.figure(4)

plt.plot(a, tv_norm ,  color='blue', linewidth=1 , marker='o' , markersize=5 )
#x1, x2, y1, y2 = plt.axis()
#plt.axis( (4.93, 30.07, y1, y2))
#plt.xticks( range(5, 31, 5) )

plt.ylabel( r'$\rho_{TV} \left ( F_{0} , \ F_{30} \right )$', fontsize=20 ) 
plt.xlabel( '$t_f$', fontsize=20 )

fig_name = 'tv_error'+ext+'.png'
plt.savefig( os.path.join( 'images' , fig_name ) , dpi=400 , bbox_inches='tight' ) 

plt.figure( 5 )

grid = np.linspace( mr.x0 , mr.x1 , len(f_true) )

mm = int( len(f_true)/2 )

plt.plot(  grid, f_true[ mm ] , color='b'  )
plt.plot(  grid, f_fit[ mm ] , color='r'  )
#plt.plot(  grid, f_init[ mm ] , color='g'  )

plt.axis([0 , 1, 0, 1.1] , fontsize=15)

plt.title('$F(x,\  y )\  \mathrm{for\  fixed}\ y$', fontsize=20)
plt.ylabel('$F(x,\  \overline{x} / 2)$', fontsize=20)
plt.xlabel('$x$', fontsize=20)
plt.xticks( [0, 1] , ('$0$',  '$\overline{x}$'), fontsize=18 )

plt.figure( 6 )

mm = -1 

plt.plot( grid,  f_true[ mm ] , color='b'  )
plt.plot( grid,  f_fit[ mm ] , color='r'  )
#plt.plot( grid,  f_init[ mm ] , color='g'  )


plt.title('$F(x,\  y )\  \mathrm{for\  fixed}\ y$', fontsize=20)
plt.ylabel('$F(x,\  \overline{x})$', fontsize=20)
plt.xlabel('$x$', fontsize=20)
plt.xticks( [0, 1] , ('$0$',  '$\overline{x}$'), fontsize=18 )
plt.axis([0 , 1, 0, 1.1] , fontsize=15)



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



fig = plt.figure(8)
ax = fig.add_subplot(111)

dimens = 1/a 
conv_error = tv_norm

ax.loglog( dimens ,  conv_error , linewidth=0, color='blue', marker='o', markersize=5 )
        

# Line in fit in loglog plot
logx = np.log( dimens )
logy = np.log( conv_error )

coeffs = np.polyfit( logx , logy , deg=1 )

lin_fit = np.poly1d( coeffs )

line_x = np.log( np.linspace( dimens[0] , dimens[-1] ) )
line_y =  np.exp( lin_fit( line_x ) ) 
ax.loglog( np.exp( line_x ) ,   line_y   , linewidth=1 , color='black')


mytext = 'Slope =$'+str( round(coeffs[0] , 1) ) + '$'

ax.text(0.7, 0.9,  mytext , style='italic',
        bbox={'facecolor':'#87CEFA', 'alpha':0.5, 'pad':10} , 
        transform=ax.transAxes, fontsize = 20 )
ax.set_ylabel( r'$\rho_{\mathrm{TV}} \left ( F_{0} , \ F_{N} \right )$', fontsize=20 ) 
ax.set_xlabel( '$1/N$', fontsize=20 )


fig_name = 'log_error'+ext+'.png'
plt.savefig( os.path.join( 'images' , fig_name ) , dpi=400 , bbox_inches='tight' ) 

        
