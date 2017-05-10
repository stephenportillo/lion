import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double
import matplotlib.pyplot as plt
import time
import astropy.wcs
import astropy.io.fits
import sys
import os
from image_eval import psf_poly_fit, image_model_eval

# ix, iy = 0. to 3.999
def testpsf(nc, cf, psf, ix, iy, lib=None):
	psf0 = image_model_eval(np.array([12.-ix/5.], dtype=np.float32), np.array([12.-iy/5.], dtype=np.float32), np.array([1.], dtype=np.float32), 0., (25,25), nc, cf, lib=lib)
	plt.subplot(2,2,1)
	plt.imshow(psf0, interpolation='none', origin='lower')
	plt.title('matrix multiply PSF')
	plt.subplot(2,2,2)
	iix = int(np.floor(ix))
	iiy = int(np.floor(iy))
	dix = ix - iix
	diy = iy - iiy
	f00 = psf[iiy:125:5,  iix:125:5]
	f01 = psf[iiy+1:125:5,iix:125:5]
	f10 = psf[iiy:125:5,  iix+1:125:5]
	f11 = psf[iiy+1:125:5,iix+1:125:5]
	realpsf = f00*(1.-dix)*(1.-diy) + f10*dix*(1.-diy) + f01*(1.-dix)*diy + f11*dix*diy
	plt.imshow(realpsf, interpolation='none', origin='lower')
	plt.title('bilinear interpolate PSF')
	invrealpsf = np.zeros((25,25))
	mask = realpsf > 1e-3
	invrealpsf[mask] = 1./realpsf[mask]
	plt.subplot(2,2,3)
	plt.title('absolute difference')
	plt.imshow(psf0-realpsf, interpolation='none', origin='lower')
	plt.colorbar()
	plt.subplot(2,2,4)
	plt.imshow((psf0-realpsf)*invrealpsf, interpolation='none', origin='lower')
	plt.colorbar()
	plt.title('fractional difference')
	plt.show()

def neighbours(x,y,neigh,i,generate=False):
	neighx = np.abs(x - x[i])
	neighy = np.abs(y - y[i])
	adjacency = np.exp(-(neighx*neighx + neighy*neighy)/(2.*neigh*neigh))
	adjacency[i] = 0.
	neighbours = np.sum(adjacency)
	if generate:
		if neighbours:
			j = np.random.choice(adjacency.size, p=adjacency.flatten()/float(neighbours))
		else:
			j = -1
		return neighbours, j
	else:
		return neighbours

dataname = sys.argv[1]
visual = int(sys.argv[2]) > 0

f = open('Data/'+dataname+'_psf.txt')
nc, nbin = [np.int32(i) for i in f.readline().split()]
f.close()
psf = np.loadtxt('Data/'+dataname+'_psf.txt', skiprows=1).astype(np.float32)
cf = psf_poly_fit(psf, nbin=nbin)
npar = cf.shape[2]
cff = cf.reshape((cf.shape[0]*cf.shape[1], cf.shape[2]))

array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
libmmult = npct.load_library('pcat-lion', '.')
libmmult.pcat_model_eval.restype = c_double
libmmult.pcat_model_eval.argtypes = [c_int, c_int, c_int, c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_1d_int, array_1d_int, array_2d_float, array_2d_float, array_2d_float]

if visual:
	testpsf(nc, cff, psf, np.float32(np.random.uniform()*4), np.float32(np.random.uniform()*4), lib=libmmult.pcat_model_eval)

f = open('Data/'+dataname+'_pix.txt')
w, h, nband = [np.int32(i) for i in f.readline().split()]
imsz = (w, h)
assert nband == 1
bias, gain = [np.float32(i) for i in f.readline().split()]
f.close()
data = np.loadtxt('Data/'+dataname+'_cts.txt').astype(np.float32)
data -= bias
trueback = np.float32(179.)
variance = data / gain
weight = 1. / variance # inverse variance
trueminf = np.float32(250.)
truealpha = np.float32(2.00)

havetruth = os.path.isfile('Data/'+dataname+'_tru.txt')
if havetruth:
	truth = np.loadtxt('Data/'+dataname+'_tru.txt')
	truex = truth[:,0]
	truey = truth[:,1]
	truef = truth[:,2]

# number of stars to use in fit
nstar = 2000
n = np.random.randint(nstar)+1
x = (np.random.uniform(size=nstar)*(imsz[0]-1)).astype(np.float32)
y = (np.random.uniform(size=nstar)*(imsz[1]-1)).astype(np.float32)
f = trueminf * np.exp(np.random.exponential(scale=1./(truealpha-1.),size=nstar)).astype(np.float32)
x[n:] = 0.
y[n:] = 0.
f[n:] = 0.
back = trueback

nsamp = 1000
nloop = 1000
nsample = np.zeros(nsamp, dtype=np.int32)
xsample = np.zeros((nsamp, nstar), dtype=np.float32)
ysample = np.zeros((nsamp, nstar), dtype=np.float32)
fsample = np.zeros((nsamp, nstar), dtype=np.float32)
acceptance = np.zeros(nsamp, dtype=np.float32)
dt1 = np.zeros(nsamp, dtype=np.float32)
dt2 = np.zeros(nsamp, dtype=np.float32)

penalty = 1.5
crad = 10
if visual:
	plt.ion()
	plt.figure(figsize=(15,5))
for j in xrange(nsamp):
	t0 = time.clock()
	nmov = np.zeros(nloop)
	movetype = np.zeros(nloop)
	accept = np.zeros(nloop)
	outbounds = np.zeros(nloop)

	resid = data.copy() # residual for zero image is data
	model, diff2 = image_model_eval(x[0:n], y[0:n], f[0:n], back, imsz, nc, cff, weights=weight, ref=resid, lib=libmmult.pcat_model_eval)
	logL = -0.5*diff2
	resid -= model

	for i in xrange(nloop):
		t1 = time.clock()
		moveweights = np.array([80., 0., 0., 30., 30., 0.])
		moveweights /= np.sum(moveweights)
		rtype = np.random.choice(moveweights.size, p=moveweights)
		movetype[i] = rtype
		# defaults
		nw = 0
		dback = np.float32(0.)
		pn = n
		factor = 0. # best way to incorporate acceptance ratio factors?
		goodmove = False
		# mover
		if rtype == 0:
			cx = np.random.uniform()*(imsz[0]-1)
			cy = np.random.uniform()*(imsz[1]-1)
			mover = np.logical_and(np.abs(cx-x) < crad, np.abs(cy-y) < crad)
			mover[n:] = False
			nw = np.sum(mover).astype(np.int32)
			df = np.random.normal(size=nw).astype(np.float32)*np.float32(60./np.sqrt(25.))
			f0 = f[mover]

			oob_flux = (df < (trueminf - f0))
			df[oob_flux] = -2*(f0[oob_flux]-trueminf) - df[oob_flux]
			pf = f0+df
			# bounce flux
			if (pf < trueminf).any():
				print f0[oob_flux], pf[oob_flux]
			# calculate flux distribution prior factor
			dlogf = np.log(pf/f0)
			factor = -truealpha*np.sum(dlogf)

			dpos_rms = np.float32(60./np.sqrt(25.))/(np.maximum(f0, pf))
			dx = np.random.normal(size=nw).astype(np.float32)*dpos_rms
			dy = np.random.normal(size=nw).astype(np.float32)*dpos_rms
			px = x[mover] + dx
			py = y[mover] + dy
			# bounce off of crad box?
			px[px < 0] = -px[px < 0]
			px[px > (imsz[0] - 1)] = 2*(imsz[0] - 1) - px[px > (imsz[0] - 1)]
			py[py < 0] = -py[py < 0]
			py[py > (imsz[1] - 1)] = 2*(imsz[1] - 1) - py[py > (imsz[1] - 1)]
			goodmove = True # always True because we bounce off the edges of the image and fmin
		# hopper
		elif rtype == 1:
			mover = np.random.uniform(size=nstar) < 4./float(n+1)
			mover[n:] = False
			nw = np.sum(mover).astype(np.int32)
			px = np.random.uniform(size=nw).astype(np.float32)*(imsz[0]-1)
			py = np.random.uniform(size=nw).astype(np.float32)*(imsz[1]-1)
			pf = f[mover]
			goodmove = (nw > 0)
		# background change
		elif rtype == 2:
			dback = np.float32(np.random.normal())
			mover = np.full(nstar, False, dtype=np.bool)
			nw = 0
			px = np.array([], dtype=np.float32)
			py = np.array([], dtype=np.float32)
			pf = np.array([], dtype=np.float32)
			goodmove = True 
		# birth and death
		elif rtype == 3:
			lifeordeath = np.random.uniform() < 1./(np.exp(penalty) + 1.)
			mover = np.full(nstar, False, dtype=np.bool)
			# birth
			if lifeordeath and n < nstar: # do not exceed n = nstar
				# append to end
				mover[n] = True
				px = np.random.uniform(size=1).astype(np.float32)*(imsz[0]-1)
				py = np.random.uniform(size=1).astype(np.float32)*(imsz[1]-1)
				pf = trueminf * np.exp(np.random.exponential(scale=1./(truealpha-1.),size=1)).astype(np.float32)
				pn = n+1
				goodmove = True
			# death
			elif not lifeordeath and n > 0: # need something to kill
				ikill = np.random.randint(n)
				mover[ikill] = True
				singlezero = np.array([0.], dtype=np.float32)
				if ikill != n-1: # put last source in killed source's place
					mover[n-1] = True
					px = np.array([x[n-1], 0], dtype=np.float32)
					py = np.array([y[n-1], 0], dtype=np.float32)
					pf = np.array([f[n-1], 0], dtype=np.float32)
				else: # or just kill the last source if we chose it
					px = singlezero
					py = singlezero
					pf = singlezero
				pn = n-1
				goodmove = True
			nw = 1
		# merges and splits
		elif rtype == 4:
			mover = np.full(nstar, False, dtype=np.bool)
			splitsville = np.random.uniform() < 1./(np.exp(penalty) + 1.)
			kickrange = 1.
			sum_f = 0
			low_n = 0
			bright_n = 0
			pn = n

			# split
			if splitsville and n > 0 and n < nstar and (f > 2*trueminf).any(): # need something to split, but don't exceed nstar
				dx = np.random.normal()*kickrange
				dy = np.random.normal()*kickrange
				bright = f > 2*trueminf
				bright_n = np.sum(bright)
				im = np.random.randint(bright_n)
				isplit = np.where(bright)[0][im]
				mover[isplit] = True
				mover[n] = True # split in place and add to end of array
				fminratio = f[isplit] / trueminf
				frac = 1./fminratio + np.random.uniform()*(1. - 2./fminratio)
				px = x[isplit] + np.array([(1-frac)*dx, -frac*dx], dtype=np.float32)
				py = y[isplit] + np.array([(1-frac)*dy, -frac*dy], dtype=np.float32)
				pf = f[isplit] * np.array([frac, 1-frac], dtype=np.float32)
				pn = n + 1

				if bright_n > 0 and (px > 0).all() and (px < imsz[0] - 1).all() and (py > 0).all() and (py < imsz[1] - 1).all() and (pf > trueminf).all():
					goodmove = True
					# need to calculate factor
					sum_f = f[isplit]
					low_n = n
					xtemp = x[0:n].copy()
					ytemp = y[0:n].copy()
					xtemp[im] = px[0]
					ytemp[im] = py[0]
					pairs = 0.5*(neighbours(np.concatenate((xtemp, px[1:2])), np.concatenate((ytemp, py[1:2])), kickrange, im) + \
						neighbours(np.concatenate((xtemp, px[1:2])), np.concatenate((ytemp, py[1:2])), kickrange, -1))
			# merge
			elif not splitsville and n > 1: # need two things to merge!
				isplit = np.random.randint(n)
				pairs, jsplit = neighbours(x[0:n], y[0:n], kickrange, isplit, generate=True) # jsplit, isplit order not guaranteed
				pairs += neighbours(x[0:n], y[0:n], kickrange, jsplit)
				pairs *= 0.5
				if jsplit != -1:
					if jsplit < isplit:
						isplit, jsplit = jsplit, isplit
					mover[isplit] = True
					mover[jsplit] = True
					sum_f = f[isplit] + f[jsplit]
					frac = f[isplit] / sum_f
					if jsplit != n-1: # merge to isplit and move last source to jsplit, only need to check jsplit because jsplit > isplit
						mover[n-1] = True
						px = np.array([frac*x[isplit]+(1-frac)*x[jsplit], x[n-1], 0], dtype=np.float32)
						py = np.array([frac*y[isplit]+(1-frac)*y[jsplit], y[n-1], 0], dtype=np.float32)
						pf = np.array([f[isplit] + f[jsplit], f[n-1], 0], dtype=np.float32)
					else: # merge to isplit, and jsplit was last source so set it to 0
						px = np.array([frac*x[isplit]+(1-frac)*y[jsplit], 0], dtype=np.float32)
						py = np.array([frac*y[isplit]+(1-frac)*y[jsplit], 0], dtype=np.float32)
						pf = np.array([f[isplit] + f[jsplit], 0], dtype=np.float32)
					low_n = n
					bright_n = np.sum(f > 2*trueminf) - np.sum(f[mover] > 2*trueminf) + np.sum(pf > 2*trueminf)
					pn = n-1
					goodmove = True # merge will be within image, and above min flux
			if goodmove:
				fminratio = sum_f / trueminf
				factor = np.log(truealpha-1) + (truealpha-1)*np.log(trueminf) - truealpha*np.log(frac*(1-frac)*sum_f) + np.log(2*np.pi*kickrange*kickrange) - np.log(imsz[0]*imsz[1]) + np.log(1. - 2./fminratio) + np.log(bright_n) - np.log(pairs) + np.log(sum_f) # last term is Jacobian
				factor *= (pn - n)
			nw = 2
		# endif rtype	
		nmov[i] = nw
		dt1[j] += time.clock() - t1

		t2 = time.clock()
		if goodmove:
			dmodel, diff2 = image_model_eval(np.concatenate((px, x[mover])), np.concatenate((py, y[mover])), np.concatenate((pf, -f[mover])), dback, imsz, nc, cff, weights=weight, ref=resid, lib=libmmult.pcat_model_eval)

			plogL = -0.5*diff2
			if np.log(np.random.uniform()) < plogL + factor - logL:
				if np.sum(mover) != px.size:
					print rtype, np.sum(mover), px, py, pf
				x[mover] = px
				y[mover] = py
				f[mover] = pf
				n = pn
				back += dback
				model += dmodel
				resid -= dmodel
				logL = plogL
				acceptance[j] += 1
				accept[i] = 1
		else:
			acceptance[j] += 0 # null move always accepted
			outbounds[i] = 1
		dt2[j] += time.clock() - t2
		
		if visual and i == 0:
			plt.clf()
			plt.subplot(1,3,1)
			plt.imshow(data, origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data), vmax=np.percentile(data, 95))
			if havetruth:
				mask = truef > 250 # will have to change this for other data sets
				plt.scatter(truex[mask], truey[mask], marker='+', s=np.sqrt(truef[mask]), color='lime')
				mask = np.logical_not(mask)
				plt.scatter(truex[mask], truey[mask], marker='+', s=np.sqrt(truef[mask]), color='g')
			plt.scatter(x[0:n], y[0:n], marker='x', s=np.sqrt(f[0:n]), color='r')
			plt.xlim(-0.5, imsz[0]-0.5)
			plt.ylim(-0.5, imsz[1]-0.5)
			plt.subplot(1,3,2)
			plt.imshow(resid*np.sqrt(weight), origin='lower', interpolation='none', cmap='bwr', vmin=-5, vmax=5)
			if j == 0:
				plt.tight_layout()
			plt.subplot(1,3,3)
			if havetruth:
				plt.hist(np.log10(truef), range=(np.log10(trueminf), np.log10(np.max(truef))), log=True, alpha=0.5, label='HST 606W', histtype='step')
				plt.hist(np.log10(f[0:n]), range=(np.log10(trueminf), np.log10(np.max(truef))), log=True, alpha=0.5, label='Chain', histtype='step')
			else:
				plt.hist(np.log10(f[0:n]), range=(np.log10(trueminf), np.ceil(np.log10(np.max(f[0:n])))), log=True, alpha=0.5, label='Chain', histtype='step')
			plt.legend()
			plt.xlabel('log10 flux')
			plt.ylim((0.5, nstar))
			plt.draw()
			plt.pause(1e-5)

		#endfor nloop
	nsample[j] = n
	xsample[j,:] = x
	ysample[j,:] = y
	fsample[j,:] = f
	acceptance[j] /= float(nloop)
	print 'Loop', j, 'background', back, 'N', n, 'proposal (ms)', dt1[j], 'likelihood (ms)', dt2[j]
	print 'nmov (mover)', np.mean(nmov[movetype == 0])
	print 'Acceptance\t(all) %0.3f (move) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(accept), np.mean(accept[movetype == 0]), np.mean(accept[movetype == 3]), np.mean(accept[movetype == 4]))
	print 'Out of bounds\t(all) %0.3f (move) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(outbounds), np.mean(outbounds[movetype == 0]), np.mean(outbounds[movetype == 3]), np.mean(outbounds[movetype == 4]))

print 'dt1 avg', np.mean(dt1), 'dt2 avg', np.mean(dt2)
print 'saving...'
np.savez('chain.npz', n=nsample, x=xsample, y=ysample, f=fsample)
