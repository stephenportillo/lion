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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from xml.dom import minidom
from scipy import interpolate

###################################

start_time = time.clock()

max_num_sources = 3000

#sets total number of iterations
iteration_number = 1

#search radius
search_radius = 0.75

gain = 0.00546689

################################### posterior sample processing #############################################

posterior_sample = np.loadtxt('run-'+sys.argv[1]+'/posterior_sample.txt')

band_num = 1

#pulls off x, y, r of each source
posterior_n_vals = posterior_sample[:,10003].astype(np.int)
posterior_x_vals = posterior_sample[:,10002+2*band_num:10002+2*band_num+max_num_sources]
posterior_y_vals = posterior_sample[:,10002+2*band_num+max_num_sources:10002+2*band_num+2*max_num_sources]
posterior_r_flux_vals = posterior_sample[:,10002+2*band_num+2*max_num_sources:10002+2*band_num+3*max_num_sources]

############################################# SORTING #######################################################

sorted_posterior_sample = np.zeros( (len(posterior_sample), max_num_sources, 3) )


start_time = time.clock()

#first, we must sort each sample in decreasing order of brightness
for i in range(0, len(posterior_sample)):

	cat = np.zeros( (max_num_sources, 3) )
	cat[:,0] = posterior_x_vals[i]
	cat[:,1] = posterior_y_vals[i]
	cat[:,2] = posterior_r_flux_vals[i]	

	cat = np.flipud( cat[cat[:,2].argsort()] )

	sorted_posterior_sample[i] = cat

print "time, sorting: " + str(time.clock() - start_time)

############################################################################################################


PCn = posterior_n_vals
PCx = sorted_posterior_sample[:,:,0]
PCy = sorted_posterior_sample[:,:,1]
PCf = sorted_posterior_sample[:,:,2]

print "Stacking catalogues..."

#creates array for KD tree creation
PCc_stack = np.zeros((np.sum(PCn), 2))
j = 0

for i in xrange(PCn.size): # don't have to for loop to stack but oh well
        n = PCn[i]
        PCc_stack[j:j+n, 0] = PCx[i, 0:n]
        PCc_stack[j:j+n, 1] = PCy[i, 0:n]
        j += n

def clusterize(seed_cat, colors, iter_num):

	#creates tree, where tree is Pcc_stack
	tree = scipy.spatial.KDTree(PCc_stack)

	#keeps track of the clusters
	clusters = np.zeros((len(posterior_sample), len(seed_cat)*3))

	#numpy mask for sources that have been matched
	mask = np.zeros(len(PCc_stack))

	#first, we iterate over all sources in the seed catalog:
	for ct in range(0, len(seed_cat)):

		#print "time elapsed, before tree " + str(ct) + ": " + str(time.clock() - start_time)

		#query ball point at seed catalog
		matches = tree.query_ball_point(seed_cat[ct], search_radius)

		#print "time elapsed, after tree " + str(ct) + ": " + str(time.clock() - start_time)

		#in each catalog, find first instance of match w/ desired source (-- first instance okay b/c every catalog is sorted brightest to faintest)
		##this for loop should naturally populate original seed catalog as well! (b/c all distances 0)
		for i in range(0, len(posterior_sample)):

			#for specific catalog, find indices of start and end within large tree
			cat_lo_ndx = np.sum(PCn[:i])
			cat_hi_ndx = np.sum(PCn[:i+1])

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
	        	                clusters[i][len(seed_cat)+ct] = y
	        	                clusters[i][2*len(seed_cat)+ct] = f


	#making sure that I recover the seed catalog exactly (good check)
	##ONLY VALID ON 1st ITERATION!!!!
	#print np.sum(clusters[cat_num][0:len(seed_cat)] - PCx[cat_num][:len(seed_cat)])
	#print np.sum(clusters[cat_num][len(seed_cat):2*len(seed_cat)] - PCy[cat_num][:len(seed_cat)])
	#print np.sum(clusters[cat_num][2*len(seed_cat):] - PCf[cat_num][:len(seed_cat)])



	#we now generate a CLASSICAL CATALOG from clusters

	cat_len = len(seed_cat)

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

	for i in range(0, len(seed_cat)):

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
		        err_mag[i] = np.absolute( ( 22.5 - 2.5*np.log10(np.percentile(f, hi)*gain) )  - ( 22.5 - 2.5*np.log10(np.percentile(f, lo)*gain) ) )

	#makes classical catalog
	classical_catalog = np.zeros( (cat_len, 9) )
	classical_catalog[:,0] = mean_x
	classical_catalog[:,1] = err_x
	classical_catalog[:,2] = mean_y
	classical_catalog[:,3] = err_y
	classical_catalog[:,4] = mean_f
	classical_catalog[:,5] = err_f
	classical_catalog[:,6] = mean_mag
	classical_catalog[:,7] = err_mag
	classical_catalog[:,8] = confidence

        #saves catalog
        np.savetxt('run-'+sys.argv[1]+'/condensed_catalog.txt', classical_catalog)

	return classical_catalog


#loads in seed catalog
dat = np.loadtxt('run-'+sys.argv[1]+'/seeds.txt')

#sets confidence cut
cut = 0.1 

#performs confidence cut
x = dat[:,0][dat[:,2] > cut*300]
y = dat[:,1][dat[:,2] > cut*300]
n = dat[:,2][dat[:,2] > cut*300]

assert x.size == y.size
assert x.size == n.size

seed_cat = np.zeros((x.size, 2))
seed_cat[:,0] = x
seed_cat[:,1] = y
cat_len = x.size

colors = []
for i in range(0, len(seed_cat)):
	colors.append(np.random.rand(3,1))

classical_catalog = clusterize(seed_cat, colors, 0)
