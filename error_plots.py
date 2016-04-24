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

pkl_file = open(os.path.join( 'data_files' , fnames[-1] ) , 'rb')

data = cPickle.load( pkl_file )        
pkl_file.close()

a = []

# Total variance norm. See Gibbs and Su for the definition.
tv_norm = np.zeros( len(data) )

for cnum in range(len(data)):
    
    a.append( data[cnum][0] )
    dx = ( mr.x1 - mr.x0 ) / data[cnum][0]
    tv_norm[cnum] = np.sum ( np.abs( data[cnum][-2]-data[cnum][-1] ) ) * (dx**2) / 2
    
a = np.asarray(a)


f_fit = data[-1][-1]
f_true = data[-1][-2]
f_init = data[-1][-3]


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

#plt.savefig( 'f_fit_beta.png', dpi=400 ) 

fig2 = plt.figure(2)

plt.title('$F_{0} (x,\ y)$', fontsize=20, y=1.04)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$y$', fontsize=20)

plt.imshow(np.flipud(f_true), interpolation='nearest', cmap=my_cmap , \
               vmin=0.001, vmax = np.max( ( np.max(f_fit) , np.max(f_true)  ) ) , extent=(0,1,0,1))
               

cbar_ax = fig2.add_axes([0.85, 0.13, 0.03, 0.75])
plt.colorbar(cax=cbar_ax)

plt.savefig( 'f_true_beta.png', dpi=400 ) 


fig3 = plt.figure(3)

plt.title('Absolute error for $F_{0}$ and $F_{30}$', fontsize=16, y=1.04)


np.seterr(all = 'ignore')

## aa = np.abs( ((gamma_fit - gamma_true ).T / np.sum(gamma_true, axis = 1) ).T )
aa =  np.abs( f_fit - f_true  )

np.seterr(all = 'raise')

aa [np.isnan(aa) ] = 0
aa [np.isinf(aa) ] = 0

imgplot = plt.imshow(np.flipud( aa )  , interpolation='nearest', cmap='Reds' , \
               vmin = 0, vmax = np.max( aa )  , extent=(0,1,0,1))


cbar_ax = fig3.add_axes([0.85, 0.13, 0.03, 0.75])
plt.colorbar(cax=cbar_ax)      
#plt.savefig('relative_error_beta.png', dpi = 400)

plt.figure(4)

plt.plot(a, tv_norm ,  color='blue', linewidth=2 )
x1, x2, y1, y2 = plt.axis()
plt.axis( (4.93, 30.07, y1, y2))
plt.xticks( range(5, 31, 5) )
plt.ylabel( '$\Vert F_{\mathrm{0}} - F_{N} \Vert_{\infty}$', fontsize=16 ) 
plt.xlabel( '$N$', fontsize=16 )
plt.title(' Error plot for the estimators $F_N$' , fontsize = 16)


## Finds and plots converging subsequence of the estimators

bb = tv_norm

lo = bb[0]

cc = np.arange(1)

pp = 1

for mm in range(1, len(tv_norm)):
           
    if bb [pp] > lo :
        bb = np.delete(bb, pp)
        
    else:
        
        lo = bb[pp]
        cc = np.append(cc, mm)
        pp = pp +1
cc = cc + 5        
plt.plot(cc, bb, 'ro', linewidth=3)

plt.figure( 5 )

grid = np.linspace( mr.x0 , mr.x1 , len(f_true) )

mm = int( data[-1][0] / 2  )

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


Gamma_fit = data[-1][-4]
Ain, Aout, Fin, Fout, nu, N, dx = mr.initialization( data[-1][0] )    

xx , yy = np.meshgrid( nu[1:] , nu[1:] )
 
 # Initial guess of gamma function for the optimization   
Gamma_true = mr.gam( xx , yy) 

