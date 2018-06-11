#alternative clustering method

import astropy.io.fits
from astropy import wcs
import numpy as np
import sys
import time
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.spatial 
# commons
import os, h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from xml.dom import minidom
from scipy import interpolate

def retr_cond(catlseed):

    #creates tree, where tree is Pcc_stack
    tree = scipy.spatial.KDTree(PCc_stack)

    #keeps track of the clusters
    clusters = np.zeros((numbsamp, len(catlseed)*3))

    #numpy mask for sources that have been matched
    mask = np.zeros(len(PCc_stack))

    #first, we iterate over all sources in the seed catalog:
    for ct in range(0, len(catlseed)):

        #print "time elapsed, before tree " + str(ct) + ": " + str(time.clock() - start_time)

        #query ball point at seed catalog
        matches = tree.query_ball_point(catlseed[ct], radisrch)

        #print "time elapsed, after tree " + str(ct) + ": " + str(time.clock() - start_time)

        #in each catalog, find first instance of match w/ desired source (-- first instance okay b/c every catalog is sorted brightest to faintest)
        ##this for loop should naturally populate original seed catalog as well! (b/c all distances 0)
        for i in range(0, len(posterior_sample)):

            #for specific catalog, find indices of start and end within large tree
            cat_lo_ndx = np.sum(posterior_n_vals[:i])
            cat_hi_ndx = np.sum(posterior_n_vals[:i+1])

            #want in the form of a numpy array so we can use array slicing/masking  
            matches = np.array(matches)

            #find the locations of matches to ct within specific catalog i
            culled_matches =  matches[np.logical_and(matches >= cat_lo_ndx, matches < cat_hi_ndx)] 

            if culled_matches.size > 0:
    
                #cut according to mask
                culled_matches = culled_matches[mask[culled_matches] == 0]
        
                #if there are matches remaining, we then find the brightest and update
                if culled_matches.size > 0:

                    #find brightest
                    match = np.min(culled_matches)

                    #flag it in the mask
                    mask[match] += 1

                    #find x, y, flux of match
                    x = PCx[i][match-cat_lo_ndx]
                    y = PCy[i][match-cat_lo_ndx]
                    f = PCf[i][match-cat_lo_ndx]

                    #add information to cluster array
                    clusters[i][ct] = x
                    clusters[i][len(catlseed)+ct] = y
                    clusters[i][2*len(catlseed)+ct] = f


    #making sure that I recover the seed catalog exactly (good check)
    ##ONLY VALID ON 1st ITERATION!!!!
    #print np.sum(clusters[cat_num][0:len(catlseed)] - PCx[cat_num][:len(catlseed)])
    #print np.sum(clusters[cat_num][len(catlseed):2*len(catlseed)] - PCy[cat_num][:len(catlseed)])
    #print np.sum(clusters[cat_num][2*len(catlseed):] - PCf[cat_num][:len(catlseed)])

    # generate condensed catalog from clusters

    cat_len = len(catlseed)

    #arrays to store 'classical' catalog parameters
    mean_x = np.zeros(cat_len)
    mean_y = np.zeros(cat_len)
    mean_f = np.zeros(cat_len)
    mean_mag = np.zeros(cat_len)
    err_x = np.zeros(cat_len)
    err_y = np.zeros(cat_len)
    err_f = np.zeros(cat_len)
    err_mag = np.zeros(cat_len)
    confidence = np.zeros(cat_len)
    
    #confidence interval defined for err_(x,y,f)
    hi = 84
    lo = 16
    for i in range(0, len(catlseed)):
        x = clusters[:,i][np.nonzero(clusters[:,i])]
        y = clusters[:,i+cat_len][np.nonzero(clusters[:,i+cat_len])]
        f = clusters[:,i+2*cat_len][np.nonzero(clusters[:,i+2*cat_len])]
        assert x.size == y.size
        assert x.size == f.size
        confidence[i] = x.size/300.0
        mean_x[i] = np.mean(x)
        mean_y[i] = np.mean(y)
        mean_f[i] = np.mean(f)
        mean_mag[i] = 22.5 - 2.5*np.log10(np.mean(f)*gain)
        if x.size > 1:
            err_x[i] = np.percentile(x, hi) - np.percentile(x, lo)
            err_y[i] = np.percentile(y, hi) - np.percentile(y, lo)
            err_f[i] = np.percentile(f, hi) - np.percentile(f, lo)
            err_mag[i] = np.absolute((22.5 - 2.5 * np.log10(np.percentile(f, hi) * gain)) - (22.5 - 2.5 * np.log10(np.percentile(f, lo) * gain)))
        pass
    classical_catalog = np.zeros((cat_len, 9))
    classical_catalog[:,0] = mean_x
    classical_catalog[:,1] = err_x
    classical_catalog[:,2] = mean_y
    classical_catalog[:,3] = err_y
    classical_catalog[:,4] = mean_f
    classical_catalog[:,5] = err_f
    classical_catalog[:,6] = mean_mag
    classical_catalog[:,7] = err_mag
    classical_catalog[:,8] = confidence
    
    # save catalog
    np.savetxt(pathcond, classical_catalog)
    
    return classical_catalog


# HACKING
#strgchan = sys.argv[1]
#strgtimestmp = strgchan[:15]
strgtimestmp = '20180607_134835'

# paths
pathlion = os.environ['LION_PATH'] + '/'
pathliondata = os.environ["LION_PATH"] + '/Data/'
pathdata = os.environ['LION_DATA_PATH'] + '/'
pathcond = pathliondata + 'cond.txt'

start_time = time.clock()

# search radius
radisrch = 0.75

# confidence cut
cut = 0.1 

# gain
gain = 0.00546689

# read the chain
print 'Reading the chain...'    
pathchan = pathliondata + strgtimestmp + '_chan.h5'
filechan = h5py.File(pathchan, 'r')
posterior_x_vals = filechan['x'][()] 
posterior_y_vals = filechan['y'][()]
posterior_r_flux_vals = filechan['f'][()]

numbsamp = len(posterior_x_vals)
posterior_n_vals = np.zeros(numbsamp, dtype=int)
indxsamp = np.arange(numbsamp)
for k in indxsamp:
    posterior_n_vals[k] = len(posterior_x_vals[k])
filechan.close()

maxmnumbsour = posterior_x_vals.shape[1]

# sort
sorted_posterior_sample = np.zeros((numbsamp, maxmnumbsour, 3))

start_time = time.clock()

# sort each sample in decreasing order of brightness
for i in range(0, numbsamp):
    cat = np.zeros((maxmnumbsour, 3))
    cat[:,0] = posterior_x_vals[i]
    cat[:,1] = posterior_y_vals[i]
    cat[:,2] = posterior_r_flux_vals[i] 
    cat = np.flipud( cat[cat[:,2].argsort()] )
    sorted_posterior_sample[i] = cat

print "time, sorting: " + str(time.clock() - start_time)

posterior_n_vals = posterior_n_vals
PCx = sorted_posterior_sample[:,:,0]
PCy = sorted_posterior_sample[:,:,1]
PCf = sorted_posterior_sample[:,:,2]

print "Stacking catalogues..."

# create array for KD tree creation
PCc_stack = np.zeros((np.sum(posterior_n_vals), 2))
j = 0

for i in xrange(posterior_n_vals.size):
        n = posterior_n_vals[i]
        PCc_stack[j:j+n, 0] = PCx[i, 0:n]
        PCc_stack[j:j+n, 1] = PCy[i, 0:n]
        j += n

# load in the seed catalog
dat = np.loadtxt(pathliondata + strgtimestmp + '_seed.txt')

# perform confidence cut
x = dat[:,0][dat[:,2] > cut*300]
y = dat[:,1][dat[:,2] > cut*300]
n = dat[:,2][dat[:,2] > cut*300]

assert x.size == y.size
assert x.size == n.size

catlseed = np.zeros((x.size, 2))
catlseed[:,0] = x
catlseed[:,1] = y
cat_len = x.size

cond = retr_cond(catlseed)
print 'cond'
print cond
print cond.shape
