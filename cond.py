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
            cat_lo_ndx = np.sum(catlnumb[:i])
            cat_hi_ndx = np.sum(catlnumb[:i+1])

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
                    x = catlsortxpos[i][match-cat_lo_ndx]
                    y = catlsortypos[i][match-cat_lo_ndx]
                    f = catlsortflux[i][match-cat_lo_ndx]

                    #add information to cluster array
                    clusters[i][ct] = x
                    clusters[i][len(catlseed)+ct] = y
                    clusters[i][2*len(catlseed)+ct] = f

    # generate condensed catalog from clusters
    numbsourseed = len(catlseed)

    #arrays to store 'classical' catalog parameters
    mean_x = np.zeros(numbsourseed)
    mean_y = np.zeros(numbsourseed)
    mean_f = np.zeros(numbsourseed)
    mean_mag = np.zeros(numbsourseed)
    err_x = np.zeros(numbsourseed)
    err_y = np.zeros(numbsourseed)
    err_f = np.zeros(numbsourseed)
    err_mag = np.zeros(numbsourseed)
    confidence = np.zeros(numbsourseed)
    
    #confidence interval defined for err_(x,y,f)
    hi = 84
    lo = 16
    for i in range(0, len(catlseed)):
        x = clusters[:,i][np.nonzero(clusters[:,i])]
        y = clusters[:,i+numbsourseed][np.nonzero(clusters[:,i+numbsourseed])]
        f = clusters[:,i+2*numbsourseed][np.nonzero(clusters[:,i+2*numbsourseed])]
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
    classical_catalog = np.zeros((numbsourseed, 9))
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
catlxpos = filechan['x'][()] 
catlypos = filechan['y'][()]
catlflux = filechan['f'][()]

numbsamp = len(catlxpos)
catlnumb = np.zeros(numbsamp, dtype=int)
indxsamp = np.arange(numbsamp)
for k in indxsamp:
    catlnumb[k] = len(catlxpos[k])
filechan.close()

maxmnumbsour = catlxpos.shape[1]

# sort the catalog in decreasing flux
catlsort = np.zeros((numbsamp, maxmnumbsour, 3))
for i in range(0, numbsamp):
    catl = np.zeros((maxmnumbsour, 3))
    catl[:,0] = catlxpos[i]
    catl[:,1] = catlypos[i]
    catl[:,2] = catlflux[i] 
    catl = np.flipud(catl[catl[:,2].argsort()])
    catlsort[i] = catl
catlsortxpos = catlsort[:,:,0]
catlsortypos = catlsort[:,:,1]
catlsortflux = catlsort[:,:,2]

print "Stacking catalogs..."

# create array for KD tree creation
PCc_stack = np.zeros((np.sum(catlnumb), 2))
j = 0
for i in xrange(catlnumb.size):
    n = catlnumb[i]
    PCc_stack[j:j+n, 0] = catlsortxpos[i, 0:n]
    PCc_stack[j:j+n, 1] = catlsortypos[i, 0:n]
    j += n

# seed catalog
## load the catalog
data = np.loadtxt(pathliondata + strgtimestmp + '_seed.txt')

## perform confidence cut
seedxpos = data[:,0][data[:,2] > cut*300]
seedypos = data[:,1][data[:,2] > cut*300]
seednumb = data[:,2][data[:,2] > cut*300]

assert seedxpos.size == seedypos.size
assert seedxpos.size == seednumb.size

catlseed = np.zeros((seedxpos.size, 2))
catlseed[:,0] = seedxpos
catlseed[:,1] = seedypos
numbsourseed = seedxpos.size

print 'catlseed'
print catlseed
print catlseed.shape

# get the condensed catalog
cond = retr_cond(catlseed)
print 'cond'
print cond
print cond.shape
