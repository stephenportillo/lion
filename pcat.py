import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double
import h5py
# in order for visual=True to work, interactive backend should be loaded before importing pyplot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import time
import astropy.wcs
import astropy.io.fits
import sys
import os
import warnings

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


def get_region(x, offsetx, regsize):
    return np.floor(x + offsetx).astype(np.int) / regsize

def idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize):
    match_x = (get_region(x[0:n], offsetx, regsize) % 2) == parity_x
    match_y = (get_region(y[0:n], offsety, regsize) % 2) == parity_y
    return np.flatnonzero(np.logical_and(match_x, match_y))

# script arguments
dataname = sys.argv[1]
visual = int(sys.argv[2]) > 0
# 1 to test, 0 not to test
testpsfn = int(sys.argv[3]) > 0
# 'star' for star only, 'stargalx' for star and galaxy
strgmode = sys.argv[4]
# 'mock' for simulated
datatype = sys.argv[5]


f = open('Data/'+dataname+'_psf.txt')
nc, nbin = [np.int32(i) for i in f.readline().split()]
f.close()
psf = np.loadtxt('Data/'+dataname+'_psf.txt', skiprows=1).astype(np.float32)
cf = psf_poly_fit(psf, nbin=nbin)
npar = cf.shape[0]

if os.path.getmtime('pcat-lion.c') > os.path.getmtime('pcat-lion.so'):
    warnings.warn('pcat-lion.c modified after compiled pcat-lion.so', Warning)

array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
array_2d_double = npct.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
libmmult = npct.load_library('pcat-lion', '.')
libmmult.pcat_model_eval.restype = None
libmmult.pcat_model_eval.argtypes = [c_int, c_int, c_int, c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_1d_int, array_1d_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]
array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")
libmmult.pcat_imag_acpt.restype = None
libmmult.pcat_imag_acpt.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_int, c_int, c_int, c_int, c_int]
#
libmmult.pcat_like_eval.restype = None
libmmult.pcat_like_eval.argtypes = [c_int, c_int, array_2d_float, array_2d_float, array_2d_float, array_2d_double, c_int, c_int, c_int, c_int]

if visual and testpsfn:
    testpsf(nc, cf, psf, np.float32(np.random.uniform()*4), np.float32(np.random.uniform()*4), lib=libmmult.pcat_model_eval)

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

print 'Lion mode:', strgmode
print 'datatype:', datatype

if datatype == 'mock':
    if strgmode == 'star':
        truth = np.loadtxt('Data/'+dataname+'_tru.txt')
        truex = truth[:,0]
        truey = truth[:,1]
        truef = truth[:,2]
    if strgmode == 'stargalx':
        pathliondata = os.environ["LION_DATA_PATH"] + '/data/'
        truth = np.loadtxt(pathliondata + 'truecnts.txt')
        filetrue = h5py.File(pathliondata + 'true.h5', 'r')
        dictglob = dict()
        for attr in filetrue:
            dictglob[attr] = filetrue[attr][()]
        filetrue.close()
    
    labldata = 'True'
else:
    labldata = 'HST 606W'

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

penalty = 1.5
regsize = 20
margin = 10
kickrange = 1.
if visual:
    plt.ion()
    plt.figure(figsize=(15,5))
for j in xrange(nsamp):
    t0 = time.clock()
    nmov = np.zeros(nloop)
    movetype = np.zeros(nloop)
    accept = np.zeros(nloop)
    outbounds = np.zeros(nloop)
    dt1 = np.zeros(nloop)
    dt2 = np.zeros(nloop)
    dt3 = np.zeros(nloop)

    offsetx = np.random.randint(regsize)
    offsety = np.random.randint(regsize)
    nregx = imsz[0] / regsize + 1
    nregy = imsz[1] / regsize + 1

    resid = data.copy() # residual for zero image is data
    model, diff2 = image_model_eval(x[0:n], y[0:n], f[0:n], back, imsz, nc, cf, weights=weight, ref=resid, lib=libmmult.pcat_model_eval, \
        regsize=regsize, margin=margin, offsetx=offsetx, offsety=offsety)
    logL = -0.5*diff2
    resid -= model

    moveweights = np.array([80., 0., 40., 40.])
    moveweights /= np.sum(moveweights)

    for i in xrange(nloop):
        t1 = time.clock()
        rtype = np.random.choice(moveweights.size, p=moveweights)
        movetype[i] = rtype
        # defaults
        nw = 0
        dback = np.float32(0.)
        pn = n
        factor = None # best way to incorporate acceptance ratio factors?
        goodmove = False

	# should regions be perturbed randomly or systematically?
	parity_x = np.random.randint(2)
	parity_y = np.random.randint(2)

	idx_move = None
	do_birth = False
	idx_kill = None
        # mover
        if rtype == 0:
            idx_move = idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize)
            nw = idx_move.size
            f0 = f.take(idx_move)

            if np.random.uniform() < 0.95:
                # linear in flux
                df = np.random.normal(size=nw).astype(np.float32)*np.float32(60./np.sqrt(25.))
                # bounce flux off of fmin
                abovefmin = f0 - trueminf
                oob_flux = (-df > abovefmin)
                df[oob_flux] = -2*abovefmin[oob_flux] - df[oob_flux]
                pf = f0+df
                # calculate flux distribution prior factor
                dlogf = np.log(pf/f0)
                factor = -truealpha*dlogf
            else:
                # logarithmic, to give bright sources a chance
                # might be bad to do and not that helpful
                dlogf = np.random.normal(size=nw).astype(np.float32)*np.float32(0.01)#/np.sqrt(25.))
                # bounce flux off of fmin
                abovefmin = np.log(f0/trueminf)
                oob_flux = (-dlogf > abovefmin)
                dlogf[oob_flux] = -2*abovefmin[oob_flux] - dlogf[oob_flux]
                pf = f0*np.exp(dlogf)
                factor = -truealpha*dlogf

            dpos_rms = np.float32(60./np.sqrt(25.))/(np.maximum(f0, pf))
            dx = np.random.normal(size=nw).astype(np.float32)*dpos_rms
            dy = np.random.normal(size=nw).astype(np.float32)*dpos_rms
            x0 = x.take(idx_move)
            y0 = y.take(idx_move)
            px = x0 + dx
            py = y0 + dy
            # bounce off of edges of image
            mask = px < 0
            px[mask] *= -1
            mask = px > (imsz[0] - 1)
            px[mask] *= -1
            px[mask] += 2*(imsz[0] - 1)
            mask = py < 0
            py[mask] *= -1
            mask = py > (imsz[1] - 1)
            py[mask] *= -1
            py[mask] += 2*(imsz[1] - 1)
            goodmove = True # always True because we bounce off the edges of the image and fmin
        # background change
        elif rtype == 1:
            dback = np.float32(np.random.normal())
            goodmove = True 
        # birth and death
        elif rtype == 2:
            lifeordeath = np.random.uniform() < 1./(np.exp(penalty) + 1.) # better to do this or put in factor?
            nbd = 9
            # birth
            if lifeordeath and n < nstar: # need room for at least one source
                nbd = min(nbd, nstar-n) # add nbd sources, or just as many as will fit
                                        # mildly violates detailed balance when n close to nstar
                # want number of regions in each direction, divided by two, rounded up
                nregx = ((imsz[0] / regsize + 1) + 1) / 2 # assumes that imsz are multiples of regsize
                nregy = ((imsz[1] / regsize + 1) + 1) / 2
                bx = ((np.random.randint(nregx, size=nbd)*2 + parity_x + np.random.uniform(size=nbd))*regsize - offsetx).astype(np.float32)
                by = ((np.random.randint(nregy, size=nbd)*2 + parity_y + np.random.uniform(size=nbd))*regsize - offsety).astype(np.float32)
                bf = trueminf * np.exp(np.random.exponential(scale=1./(truealpha-1.),size=nbd)).astype(np.float32)

		# some sources might be generated outside image
		inbounds = (bx > 0) * (bx < (imsz[0] -1)) * (by > 0) * (by < imsz[1] - 1)
		idx_in = np.flatnonzero(inbounds)
                nw = idx_in.size
		bx = bx.take(idx_in)
                by = by.take(idx_in)
                bf = bf.take(idx_in)
                do_birth = True
                goodmove = True
            # death
            # does region based death obey detailed balance?
            elif not lifeordeath and n > 0: # need something to kill
                idx_reg = idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize)

		nbd = min(nbd, idx_reg.size) # kill nbd sources, or however many sources remain
                nw = nbd
                # need to handle case where nbd = 0?
                idx_kill = np.random.choice(idx_reg, size=nbd, replace=False)
		xk = x.take(idx_kill)
                yk = y.take(idx_kill)
                fk = f.take(idx_kill)
                goodmove = True
        # merges and splits
        elif rtype == 3:
            splitsville = np.random.uniform() < 1./(np.exp(penalty) + 1.)
            idx_reg = idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize)
            sum_f = 0
            low_n = 0
            idx_bright = idx_reg.take(np.flatnonzero(f.take(idx_reg) > 2*trueminf)) # in region!
            bright_n = idx_bright.size

            nms = 9
            # split
            if splitsville and n > 0 and n < nstar and bright_n > 0: # need something to split, but don't exceed nstar
                nms = min(nms, bright_n, nstar-n) # need bright source AND room for split source
                dx = (np.random.normal(size=nms)*kickrange).astype(np.float32)
                dy = (np.random.normal(size=nms)*kickrange).astype(np.float32)
		idx_move = np.random.choice(idx_bright, size=nms, replace=False)
                x0 = x.take(idx_move)
                y0 = y.take(idx_move)
		f0 = f.take(idx_move)
                fminratio = f0 / trueminf
                frac = (1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32)
                px = x0 + ((1-frac)*dx)
                py = y0 + ((1-frac)*dy)
                pf = f0 * frac
                do_birth = True
                bx = x0 - frac*dx
                by = y0 - frac*dy
                bf = f0 * (1-frac)

                # don't want to think about how to bounce split-merge
                # don't need to check if above fmin, because of how frac is decided
                inbounds = (px > 0) * (px < imsz[0] - 1) * (py > 0) * (py < imsz[1] - 1) * \
                           (bx > 0) * (bx < imsz[0] - 1) * (by > 0) * (by < imsz[1] - 1)
                idx_in = np.flatnonzero(inbounds)
                x0 = x0.take(idx_in)
                y0 = y0.take(idx_in)
                f0 = f0.take(idx_in)
                px = px.take(idx_in)
                py = py.take(idx_in)
                pf = pf.take(idx_in)
                bx = bx.take(idx_in)
                by = by.take(idx_in)
                bf = bf.take(idx_in)
                idx_move = idx_move.take(idx_in)
                fminratio = fminratio.take(idx_in)
                frac = frac.take(idx_in)
                goodmove = idx_in.size > 0

                # need to calculate factor
                sum_f = f0
                nms = idx_in.size
                nw = nms
                pairs = np.zeros(nms)
                for k in xrange(nms):
                    xtemp = x[0:n].copy()
                    ytemp = y[0:n].copy()
                    xtemp[idx_move[k]] = px[k]
                    ytemp[idx_move[k]] = py[k]
                    xtemp = np.concatenate([xtemp, bx[k:k+1]])
                    ytemp = np.concatenate([ytemp, by[k:k+1]])

                    pairs[k] =  neighbours(xtemp, ytemp, kickrange, idx_move[k])
                    pairs[k] += neighbours(xtemp, ytemp, kickrange, n)
                pairs *= 0.5
            # merge
            elif not splitsville and idx_reg.size > 1: # need two things to merge!
                nms = min(nms, idx_reg.size/2)
                idx_move = np.zeros(nms, dtype=np.int)
                idx_kill = np.zeros(nms, dtype=np.int)
                choosable = np.zeros(nstar, dtype=np.bool)
                choosable[idx_reg] = True
                nchoosable = float(np.count_nonzero(choosable))
                pairs = np.zeros(nms)

                for k in xrange(nms):
                    idx_move[k] = np.random.choice(nstar, p=choosable/nchoosable)
                    pairs[k], idx_kill[k] = neighbours(x[0:n], y[0:n], kickrange, idx_move[k], generate=True)
                    # prevent sources from being involved in multiple proposals
                    if not choosable[idx_kill[k]]:
                        idx_kill[k] = -1
                    if idx_kill[k] != -1:
                        pairs[k] += neighbours(x[0:n], y[0:n], kickrange, idx_kill[k])
                        choosable[idx_move[k]] = False
                        choosable[idx_kill[k]] = False
                        nchoosable -= 2
                pairs *= 0.5

                inbounds = (idx_kill != -1)
                idx_in = np.flatnonzero(inbounds)
                nms = idx_in.size
                nw = nms
                idx_move = idx_move.take(idx_in)
                idx_kill = idx_kill.take(idx_in)
                pairs = pairs.take(idx_in)
                goodmove = idx_in.size > 0

                x0 = x.take(idx_move)
                y0 = y.take(idx_move)
                f0 = f.take(idx_move)
                xk = x.take(idx_kill)
                yk = y.take(idx_kill)
                fk = f.take(idx_kill)
                sum_f = f0 + fk
                fminratio = sum_f / trueminf
                frac = f0 / sum_f
                px = frac*x0 + (1-frac)*xk
                py = frac*y0 + (1-frac)*yk
                pf = f0 + fk
                # turn bright_n into an array
                bright_n = bright_n - (f0 > 2*trueminf) - (fk > 2*trueminf) + (pf > 2*trueminf)
            if goodmove:
                factor = np.log(truealpha-1) + (truealpha-1)*np.log(trueminf) - truealpha*np.log(frac*(1-frac)*sum_f) + np.log(2*np.pi*kickrange*kickrange) - np.log(imsz[0]*imsz[1]) + np.log(1. - 2./fminratio) + np.log(bright_n) - np.log(pairs) + np.log(sum_f) # last term is Jacobian
                if not splitsville:
                    factor *= -1
        # endif rtype   
        nmov[i] = nw
        dt1[i] = time.clock() - t1

        if goodmove:
            t2 = time.clock()
            xtemp = []
            ytemp = []
            ftemp = []
            if idx_move is not None:
                xtemp.extend([ x0, px])
                ytemp.extend([ y0, py])
                ftemp.extend([-f0, pf])
            if do_birth:
                xtemp.append(bx)
                ytemp.append(by)
                ftemp.append(bf)
            if idx_kill is not None:
                xtemp.append(xk)
                ytemp.append(yk)
                ftemp.append(-fk)

            dmodel, diff2 = image_model_eval(np.concatenate(xtemp), np.concatenate(ytemp), np.concatenate(ftemp), dback, imsz, nc, cf, weights=weight, ref=resid, lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=offsetx, offsety=offsety)
            plogL = -0.5*diff2
            dt2[i] = time.clock() - t2

            t3 = time.clock()
            nregx = imsz[0] / regsize + 1
            nregy = imsz[1] / regsize + 1
            refx = None
            refy = None
            if idx_move is not None:
                refx = x0
                refy = y0
            else:
                if do_birth:
                    refx = bx
                    refy = by
                if idx_kill is not None:
                    refx = xk
                    refy = yk
            regionx = get_region(refx, offsetx, regsize)
            regiony = get_region(refy, offsety, regsize)

            plogL[(1-parity_y)::2,:] = float('-inf') # don't accept off-parity regions
            plogL[:,(1-parity_x)::2] = float('-inf')
            dlogP = plogL - logL
            if factor is not None:
                dlogP[regiony, regionx] += factor
            acceptreg = (np.log(np.random.uniform(size=(nregy, nregx))) < dlogP).astype(np.int32)
            acceptprop = acceptreg[regiony, regionx]
            numaccept = np.count_nonzero(acceptprop)

            # only keep dmodel in accepted regions+margins
            dmodel_acpt = np.zeros_like(dmodel)
            libmmult.pcat_imag_acpt(imsz[0], imsz[1], dmodel, dmodel_acpt, acceptreg, regsize, margin, offsetx, offsety)
            # using this dmodel containing only accepted moves, update logL
            diff2.fill(0)
            libmmult.pcat_like_eval(imsz[0], imsz[1], dmodel_acpt, resid, weight, diff2, regsize, margin, offsetx, offsety)
            logL = -0.5*diff2
            resid -= dmodel_acpt # has to occur after pcat_like_eval, because resid is used as ref
            model += dmodel_acpt
            # implement accepted moves
            if idx_move is not None:
                px_a = px.compress(acceptprop)
                py_a = py.compress(acceptprop)
                pf_a = pf.compress(acceptprop)
                idx_move_a = idx_move.compress(acceptprop)
                x[idx_move_a] = px_a
                y[idx_move_a] = py_a
                f[idx_move_a] = pf_a
            if do_birth:
                bx_a = bx.compress(acceptprop)
                by_a = by.compress(acceptprop)
                bf_a = bf.compress(acceptprop)
                num_born = np.count_nonzero(acceptprop)
                x[n:n+num_born] = bx_a
                y[n:n+num_born] = by_a
                f[n:n+num_born] = bf_a
                n += num_born
            if idx_kill is not None:
                idx_kill_a = idx_kill.compress(acceptprop)
                num_kill = idx_kill_a.size
                x[0:nstar-num_kill] = np.delete(x, idx_kill_a)
                y[0:nstar-num_kill] = np.delete(y, idx_kill_a)
                f[0:nstar-num_kill] = np.delete(f, idx_kill_a)
                x[nstar-num_kill:] = 0
                y[nstar-num_kill:] = 0
                f[nstar-num_kill:] = 0
                n -= num_kill
            dt3[i] = time.clock() - t3

            # hmm...
            #back += dback
            if acceptprop.size > 0: 
                accept[i] = np.count_nonzero(acceptprop) / float(acceptprop.size)
            else:
                accept[i] = 0
        else:
            outbounds[i] = 1
    
        if visual and i == 0:
            plt.figure(1)
            plt.clf()
            plt.subplot(1,3,1)
            plt.imshow(data, origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data), vmax=np.percentile(data, 95))
            if datatype == 'mock':
                if strgmode == 'star':
                    mask = truef > 250 # will have to change this for other data sets
                    plt.scatter(truex[mask], truey[mask], marker='+', s=np.sqrt(truef[mask]), color='lime')
                    mask = np.logical_not(mask)
                    plt.scatter(truex[mask], truey[mask], marker='+', s=np.sqrt(truef[mask]), color='g')
                else:
                    plt.scatter(dictglob['truexposstar'], truey[mask], marker='+', s=np.sqrt(truef[mask]), color='g')
            plt.scatter(x[0:n], y[0:n], marker='x', s=np.sqrt(f[0:n]), color='r')
            plt.xlim(-0.5, imsz[0]-0.5)
            plt.ylim(-0.5, imsz[1]-0.5)
            plt.subplot(1,3,2)
            plt.imshow(resid*np.sqrt(weight), origin='lower', interpolation='none', cmap='bwr', vmin=-5, vmax=5)
            if j == 0:
                plt.tight_layout()
            plt.subplot(1,3,3)

            if datatype == 'mock':
                plt.hist(np.log10(truef), range=(np.log10(trueminf), np.log10(np.max(truef))), log=True, alpha=0.5, label=labldata, histtype='step')
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
    print 'Loop', j, 'background', back, 'N', n
    print 'Acceptance\t(all) %0.3f (move) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(accept), np.mean(accept[movetype == 0]), np.mean(accept[movetype == 2]), np.mean(accept[movetype == 3]))
    print 'Out of bounds\t(all) %0.3f (move) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(outbounds), np.mean(outbounds[movetype == 0]), np.mean(outbounds[movetype == 2]), np.mean(outbounds[movetype == 3]))
    print '# src pert\t(all) %0.1f (move) %0.1f (B-D) %0.1f (M-S) %0.1f' % (np.mean(nmov), np.mean(nmov[movetype == 0]), np.mean(nmov[movetype == 2]), np.mean(nmov[movetype == 3]))
    print '-'*16
    dt1 *= 1000
    dt2 *= 1000
    dt3 *= 1000
    print 'Proposal (ms)\t(all) %0.3f (move) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(dt1), np.mean(dt1[movetype == 0]) , np.mean(dt1[movetype == 2]), np.mean(dt1[movetype == 3]))
    print 'Likelihood (ms)\t(all) %0.3f (move) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(dt2), np.mean(dt2[movetype == 0]) , np.mean(dt2[movetype == 2]), np.mean(dt2[movetype == 3]))
    print 'Implement (ms)\t(all) %0.3f (move) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(dt3), np.mean(dt3[movetype == 0]) , np.mean(dt3[movetype == 2]), np.mean(dt3[movetype == 3]))
    print '='*16

print 'saving...'
np.savez('chain.npz', n=nsample, x=xsample, y=ysample, f=fsample)
