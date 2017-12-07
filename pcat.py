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
from galaxy import to_moments, from_moments, retr_sers, retr_tranphon

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
    oldadj = adjacency.copy()
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
    testpsf(nc, cf, psf, np.float32(np.random.uniform()*4), np.float32(np.random.uniform()*4), libmmult.pcat_model_eval)

f = open('Data/'+dataname+'_pix.txt')
w, h, nband = [np.int32(i) for i in f.readline().split()]
imsz = (w, h)
assert nband == 1
bias, gain = [np.float32(i) for i in f.readline().split()]
f.close()
data = np.loadtxt('Data/'+dataname+'_cts.txt').astype(np.float32)
data -= bias
trueback = np.float32(445*250)#179.)
variance = data / gain
weight = 1. / variance # inverse variance

print 'Lion mode:', strgmode
print 'datatype:', datatype

if datatype == 'mock':
    if strgmode == 'star':
        truth = np.loadtxt('Data/'+dataname+'_tru.txt')
        truex = truth[:,0]
        truey = truth[:,1]
        truef = truth[:,2]
    if strgmode == 'galx':
        truth_s = np.loadtxt('Data/'+dataname+'_str.txt')
        truex = truth_s[:,0]
        truey = truth_s[:,1]
        truef = truth_s[:,2]
        truth_g = np.loadtxt('Data/'+dataname+'_gal.txt')
        truexg = truth_g[:,0]
        trueyg = truth_g[:,1]
        truefg = truth_g[:,2]
        truexxg= truth_g[:,3]
        truexyg= truth_g[:,4]
        trueyyg= truth_g[:,5]
        truerng, theta, phi = from_moments(truexxg, truexyg, trueyyg)
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

ntemps = 1
temps = np.sqrt(2) ** np.arange(ntemps)

regsize = 20
assert imsz[0] % regsize == 0
assert imsz[1] % regsize == 0
margin = 10

class Proposal:
    gridphon, amplphon = retr_sers(sersindx=2.)
    _X = 0
    _Y = 1
    _F = 2
    _XX = 3
    _XY = 4
    _YY = 5

    def __init__(self):
        self.idx_move = None
        self.idx_move_g = None
        self.do_birth = False
        self.do_birth_g = False
        self.idx_kill = None
        self.idx_kill_g = None
        self.factor = None
        self.goodmove = False

        self.xphon = np.array([], dtype=np.float32)
        self.yphon = np.array([], dtype=np.float32)
        self.fphon = np.array([], dtype=np.float32)

    def set_factor(self, factor):
        self.factor = factor

    def assert_types(self):
        assert self.xphon.dtype == np.float32
        assert self.yphon.dtype == np.float32
        assert self.fphon.dtype == np.float32

    def __add_phonions_stars(self, stars, remove=False):
        fluxmult = -1 if remove else 1
        self.xphon = np.append(self.xphon, stars[self._X,:])
        self.yphon = np.append(self.yphon, stars[self._Y,:])
        self.fphon = np.append(self.fphon, fluxmult*stars[self._F,:])
        self.assert_types()

    def __add_phonions_galaxies(self, galaxies, remove=False):
        fluxmult = -1 if remove else 1
        xtemp, ytemp, ftemp = retr_tranphon(self.gridphon, self.amplphon, galaxies)
        self.xphon = np.append(self.xphon, xtemp)
        self.yphon = np.append(self.yphon, ytemp)
        self.fphon = np.append(self.fphon, fluxmult*ftemp)
        self.assert_types()

    def add_move_stars(self, idx_move, stars0, starsp):
        self.idx_move = idx_move
        self.stars0 = stars0
        self.starsp = starsp
        self.goodmove = True
        self.__add_phonions_stars(stars0, remove=True)
        self.__add_phonions_stars(starsp)

    def add_birth_stars(self, starsb):
        self.do_birth = True
        self.starsb = starsb
        self.goodmove = True
        if starsb.ndim == 3:
            starsb = starsb.reshape((starsb.shape[0], starsb.shape[1]*starsb.shape[2]))
        self.__add_phonions_stars(starsb)

    def add_death_stars(self, idx_kill, starsk):
        self.idx_kill = idx_kill
        self.starsk = starsk
        self.goodmove = True
        if starsk.ndim == 3:
            starsk = starsk.reshape((starsk.shape[0], starsk.shape[1]*starsk.shape[2]))
        self.__add_phonions_stars(starsk, remove=True)

    def add_move_galaxies(self, idx_move_g, galaxies0, galaxiesp):
        self.idx_move_g = idx_move_g
        self.galaxies0 = galaxies0
        self.galaxiesp = galaxiesp
        self.goodmove = True
        self.__add_phonions_galaxies(galaxies0, remove=True)
        self.__add_phonions_galaxies(galaxiesp)

    def add_birth_galaxies(self, galaxiesb):
        self.do_birth_g = True
        self.galaxiesb = galaxiesb
        self.goodmove = True
        self.__add_phonions_galaxies(galaxiesb)

    def add_death_galaxies(self, idx_kill_g, galaxiesk):
        self.idx_kill_g = idx_kill_g
        self.galaxiesk = galaxiesk
        self.goodmove = True
        self.__add_phonions_galaxies(galaxiesk, remove=True)

    def get_ref_xy(self):
        if self.idx_move is not None:
            return self.stars0[self._X,:], self.stars0[self._Y,:]
        elif self.idx_move_g is not None:
            return self.galaxies0[self._X,:], self.galaxies0[self._Y,:]
        elif self.do_birth:
            bx, by = self.starsb[[self._X,self._Y],:]
            refx = bx if bx.ndim == 1 else bx[:,0]
            refy = by if by.ndim == 1 else by[:,0]
            return refx, refy
        elif self.do_birth_g:
            return self.galaxiesb[self._X,:], self.galaxiesb[self._Y,:]
        elif self.idx_kill is not None:
            xk, yk = self.starsk[[self._X,self._Y],:]
            refx = xk if xk.ndim == 1 else xk[:,0]
            refy = yk if yk.ndim == 1 else yk[:,0]
            return refx, refy
        elif self.idx_kill_g is not None:
            return self.galaxiesk[self._X,:], self.galaxiesk[self._Y,:]

class Model:
    # should these be class or instance variables?
    nstar = 100
    trueminf = np.float32(250.*136)
    truealpha = np.float32(2.00)
    back = trueback

    ngalx = 100
    trueminf_g = np.float32(250.*136)
    truealpha_g = np.float32(2.00)
    truermin_g = np.float32(1.00)

    gridphon, amplphon = retr_sers(sersindx=2.)

    penalty = 1.5
    penalty_g = 3.0
    kickrange = 1.
    kickrange_g = 1.

    _X = 0
    _Y = 1
    _F = 2
    _XX = 3
    _XY = 4
    _YY = 5

    def __init__(self):
        self.n = np.random.randint(self.nstar)+1
        self.stars = np.zeros((3,self.nstar), dtype=np.float32)
        self.stars[:,0:self.n] = np.random.uniform(size=(3,self.n))  # refactor into some sort of prior function?
        self.stars[self._X,0:self.n] *= imsz[0]-1
        self.stars[self._Y,0:self.n] *= imsz[1]-1
        self.stars[self._F,0:self.n] **= -1./(self.truealpha - 1.)
        self.stars[self._F,0:self.n] *= self.trueminf

        self.ng = np.random.randint(self.ngalx)+1
        self.galaxies = np.zeros((6,self.ngalx), dtype=np.float32)
        self.galaxies[[self._X,self._Y,self._F],0:self.ng] = np.random.uniform(size=(3,self.ng))
        self.galaxies[self._X,0:self.ng] *= imsz[0]-1
        self.galaxies[self._Y,0:self.ng] *= imsz[1]-1
        self.galaxies[self._F,0:self.ng] **= -1./(self.truealpha_g - 1.)
        self.galaxies[self._F,0:self.ng] *= self.trueminf_g
        self.galaxies[[self._XX,self._XY,self._YY],0:self.ng] = self.moments_from_prior(self.truermin_g, self.ng)

    # should offsetx/y, parity_x/y be instance variables?

    def moments_from_prior(self, truermin_g, ngalx, slope=np.float32(4)):
        rg = truermin_g*np.exp(np.random.exponential(scale=1./(slope-1.),size=ngalx)).astype(np.float32)
        ug = np.random.uniform(low=3e-4, high=1., size=ngalx).astype(np.float32) #3e-4 for numerics
        thetag = np.arccos(ug).astype(np.float32)
        phig = (np.random.uniform(size=ngalx)*np.pi - np.pi/2.).astype(np.float32)
        return to_moments(rg, thetag, phig)

    def log_prior_moments(self, galaxies):
        xx, xy, yy = galaxies[[self._XX, self._XY, self._YY],:]
        slope = 4.
        a, theta, phi = from_moments(xx, xy, yy)
        u = np.cos(theta)
        return np.log(slope-1) + (slope-1)*np.log(self.truermin_g) - slope*np.log(a) - 5*np.log(a) - np.log(u*u) - np.log(1-u*u)

    def run_sampler(self, temperature, nloop=1000, visual=False):
        t0 = time.clock()
        nmov = np.zeros(nloop)
        movetype = np.zeros(nloop)
        accept = np.zeros(nloop)
        outbounds = np.zeros(nloop)
        dt1 = np.zeros(nloop)
        dt2 = np.zeros(nloop)
        dt3 = np.zeros(nloop)

        self.offsetx = np.random.randint(regsize)
        self.offsety = np.random.randint(regsize)
        self.nregx = imsz[0] / regsize + 1
        self.nregy = imsz[1] / regsize + 1

        resid = data.copy() # residual for zero image is data
        if strgmode == 'star':
            evalx = self.stars[self._X,0:self.n]
            evaly = self.stars[self._Y,0:self.n]
            evalf = self.stars[self._F,0:self.n]
        else:
            xposphon, yposphon, specphon = retr_tranphon(self.gridphon, self.amplphon, self.galaxies[:,0:self.ng])
            evalx = np.concatenate([self.stars[self._X,0:self.n], xposphon]).astype(np.float32)
            evaly = np.concatenate([self.stars[self._Y,0:self.n], yposphon]).astype(np.float32)
            evalf = np.concatenate([self.stars[self._F,0:self.n], specphon]).astype(np.float32)
        n_phon = evalx.size
        model, diff2 = image_model_eval(evalx, evaly, evalf, self.back, imsz, nc, cf, weights=weight, ref=resid, lib=libmmult.pcat_model_eval, \
            regsize=regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety)
        logL = -0.5*diff2
        resid -= model

        moveweights = np.array([80., 40., 40.])
        if strgmode == 'galx':
            moveweights = np.array([80., 40., 40., 80., 40., 40., 40., 40., 40.])
        moveweights /= np.sum(moveweights)

        for i in xrange(nloop):
            t1 = time.clock()
            rtype = np.random.choice(moveweights.size, p=moveweights)
            movetype[i] = rtype
            # defaults
            dback = np.float32(0.)
            pn = self.n

            # should regions be perturbed randomly or systematically?
            self.parity_x = np.random.randint(2)
            self.parity_y = np.random.randint(2)

            movetypes = ['P *', 'BD *', 'MS *', 'P g', 'BD g', '*-g', '**-g', '*g-g', 'MS g']
            movefns = [self.move_stars, self.birth_death_stars, self.merge_split_stars, self.move_galaxies, self.birth_death_galaxies, self.star_galaxy, \
                    self.twostars_galaxy, self.stargalaxy_galaxy, self.merge_split_galaxies]
            proposal = movefns[rtype]()

            dt1[i] = time.clock() - t1

            if proposal.goodmove:
                t2 = time.clock()
                dmodel, diff2 = image_model_eval(proposal.xphon, proposal.yphon, proposal.fphon, dback, imsz, nc, cf, weights=weight, ref=resid, lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety)
                plogL = -0.5*diff2
                dt2[i] = time.clock() - t2

                t3 = time.clock()
                refx, refy = proposal.get_ref_xy()
                regionx = get_region(refx, self.offsetx, regsize)
                regiony = get_region(refy, self.offsety, regsize)

                plogL[(1-self.parity_y)::2,:] = float('-inf') # don't accept off-parity regions
                plogL[:,(1-self.parity_x)::2] = float('-inf')
                dlogP = (plogL - logL) / temperature
                if proposal.factor is not None:
                    dlogP[regiony, regionx] += proposal.factor
                acceptreg = (np.log(np.random.uniform(size=(self.nregy, self.nregx))) < dlogP).astype(np.int32)
                acceptprop = acceptreg[regiony, regionx]
                numaccept = np.count_nonzero(acceptprop)

                # only keep dmodel in accepted regions+margins
                dmodel_acpt = np.zeros_like(dmodel)
                libmmult.pcat_imag_acpt(imsz[0], imsz[1], dmodel, dmodel_acpt, acceptreg, regsize, margin, self.offsetx, self.offsety)
                # using this dmodel containing only accepted moves, update logL
                diff2.fill(0)
                libmmult.pcat_like_eval(imsz[0], imsz[1], dmodel_acpt, resid, weight, diff2, regsize, margin, self.offsetx, self.offsety)
                logL = -0.5*diff2
                resid -= dmodel_acpt # has to occur after pcat_like_eval, because resid is used as ref
                model += dmodel_acpt
                # implement accepted moves
                if proposal.idx_move is not None:
                    starsp = proposal.starsp.compress(acceptprop, axis=1)
                    idx_move_a = proposal.idx_move.compress(acceptprop)
                    self.stars[:, idx_move_a] = starsp
                if proposal.idx_move_g is not None:
                    galaxiesp = proposal.galaxiesp.compress(acceptprop, axis=1)
                    idx_move_a = proposal.idx_move_g.compress(acceptprop)
                    self.galaxies[:, idx_move_a] = galaxiesp
                if proposal.do_birth:
                    starsb = proposal.starsb.compress(acceptprop, axis=1)
                    starsb = starsb.reshape((3,-1))
                    num_born = starsb.shape[1]
                    self.stars[:, self.n:self.n+num_born] = starsb
                    self.n += num_born
                if proposal.do_birth_g:
                    galaxiesb = proposal.galaxiesb.compress(acceptprop, axis=1)
                    num_born = galaxiesb.shape[1]
                    self.galaxies[:, self.ng:self.ng+num_born] = galaxiesb
                    self.ng += num_born
                if proposal.idx_kill is not None:
                    idx_kill_a = proposal.idx_kill.compress(acceptprop, axis=0).flatten()
                    num_kill = idx_kill_a.size
                    # nstar is correct, not n, because x,y,f are full nstar arrays
                    self.stars[:, 0:self.nstar-num_kill] = np.delete(self.stars, idx_kill_a, axis=1)
                    self.stars[:, self.nstar-num_kill:] = 0
                    self.n -= num_kill
                if proposal.idx_kill_g is not None:
                    idx_kill_a = proposal.idx_kill_g.compress(acceptprop)
                    num_kill = idx_kill_a.size
                    # like above, ngalx is correct
                    self.galaxies[:, 0:self.ngalx-num_kill] = np.delete(self.galaxies, idx_kill_a, axis=1)
                    self.galaxies[:, self.ngalx-num_kill:] = 0
                    self.ng -= num_kill
                dt3[i] = time.clock() - t3

                # hmm...
                #back += dback
                if acceptprop.size > 0: 
                    accept[i] = np.count_nonzero(acceptprop) / float(acceptprop.size)
                else:
                    accept[i] = 0
            else:
                outbounds[i] = 1

        chi2 = np.sum(weight*(data-model)*(data-model))
        fmtstr = '\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f (Pg) %0.3f (BDg) %0.3f (S-g) %0.3f (gSg) %0.3f (gMS) %0.3f'
        print 'Temperature', temperature, 'background', self.back, 'N_star', self.n, 'N_gal', self.ng, 'N_phon', n_phon, 'chi^2', chi2
        dt1 *= 1000
        dt2 *= 1000
        dt3 *= 1000
        statlabels = ['Acceptance', 'Out of Bounds', 'Proposal (ms)', 'Likelihood (ms)', 'Implement (ms)']
        statarrays = [accept, outbounds, dt1, dt2, dt3]
        for j in xrange(len(statlabels)):
            print statlabels[j]+'\t(all) %0.3f' % (np.mean(statarrays[j])),
            for k in xrange(len(movetypes)):
                print '('+movetypes[k]+') %0.3f' % (np.mean(statarrays[j][movetype == k])),
            print
            if j == 1:
                print '-'*16
        print '='*16

        if visual:
                plt.figure(1)
                plt.clf()
                plt.subplot(1,3,1)
                plt.imshow(data, origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data), vmax=np.percentile(data, 95))
                sizefac = 10.*136
                if datatype == 'mock':
                    if strgmode == 'star' or strgmode == 'galx':
                        mask = truef > 250 # will have to change this for other data sets
                        plt.scatter(truex[mask], truey[mask], marker='+', s=truef[mask] / sizefac, color='lime')
                        mask = np.logical_not(mask)
                        plt.scatter(truex[mask], truey[mask], marker='+', s=truef[mask] / sizefac, color='g')
                    if strgmode == 'galx':
                        plt.scatter(truexg, trueyg, marker='1', s=truefg / sizefac, color='lime')
                        plt.scatter(truexg, trueyg, marker='o', s=truerng*truerng*4, edgecolors='lime', facecolors='none')
                    if strgmode == 'stargalx':
                        plt.scatter(dictglob['truexposstar'], truey[mask], marker='+', s=np.sqrt(truef[mask]), color='g')
                plt.scatter(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], marker='x', s=self.stars[self._F, 0:self.n]/sizefac, color='r')
                if strgmode == 'galx':
                    plt.scatter(self.galaxies[self._X, 0:self.ng], self.galaxies[self._Y, 0:self.ng], marker='2', s=self.galaxies[self._F, 0:self.ng]/sizefac, color='r')
                    a, theta, phi = from_moments(self.galaxies[self._XX, 0:self.ng], self.galaxies[self._XY, 0:self.ng], self.galaxies[self._YY, 0:self.ng])
                    plt.scatter(self.galaxies[self._X, 0:self.ng], self.galaxies[self._Y, 0:self.ng], marker='o', s=4*a*a, edgecolors='red', facecolors='none')
                plt.xlim(-0.5, imsz[0]-0.5)
                plt.ylim(-0.5, imsz[1]-0.5)
                plt.subplot(1,3,2)
                plt.imshow(resid*np.sqrt(weight), origin='lower', interpolation='none', cmap='bwr', vmin=-5, vmax=5)
                if j == 0:
                    plt.tight_layout()
                plt.subplot(1,3,3)

                if datatype == 'mock':
                    plt.hist(np.log10(truef), range=(np.log10(self.trueminf), np.log10(np.max(truef))), log=True, alpha=0.5, label=labldata, histtype='step')
                    plt.hist(np.log10(self.stars[self._F, 0:self.n]), range=(np.log10(self.trueminf), np.log10(np.max(truef))), log=True, alpha=0.5, label='Chain', histtype='step')
                else:
                    plt.hist(np.log10(self.stars[self._F, 0:self.n]), range=(np.log10(self.trueminf), np.ceil(np.log10(np.max(self.f[0:self.n])))), log=True, alpha=0.5, label='Chain', histtype='step')
                plt.legend()
                plt.xlabel('log10 flux')
                plt.ylim((0.5, self.nstar))
                plt.draw()
                plt.pause(1e-5)


        return self.n, self.ng, chi2

    def idx_parity_stars(self):
        return idx_parity(self.stars[self._X,:], self.stars[self._Y,:], self.n, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize)

    def idx_parity_galaxies(self):
        return idx_parity(self.galaxies[self._X,:], self.galaxies[self._Y,:], self.ng, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize)

    def bounce_off_edges(self, catalogue): # works on both stars and galaxies
        mask = catalogue[self._X,:] < 0
        catalogue[self._X, mask] *= -1
        mask = catalogue[self._X,:] > (imsz[0] - 1)
        catalogue[self._X, mask] *= -1
        catalogue[self._X, mask] += 2*(imsz[0] - 1)
        mask = catalogue[self._Y,:] < 0
        catalogue[self._Y, mask] *= -1
        mask = catalogue[self._Y,:] > (imsz[1] - 1)
        catalogue[self._Y, mask] *= -1
        catalogue[self._Y, mask] += 2*(imsz[1] - 1)
        # these are all inplace operations, so no return value

    def in_bounds(self, catalogue):
        return np.logical_and(np.logical_and(catalogue[self._X,:] > 0, catalogue[self._X,:] < (imsz[0] -1)), \
                np.logical_and(catalogue[self._Y,:] > 0, catalogue[self._Y,:] < imsz[1] - 1))

    def move_stars(self): 
        idx_move = self.idx_parity_stars()
        nw = idx_move.size
        stars0 = self.stars.take(idx_move, axis=1)
        starsp = np.empty_like(stars0)
        f0 = stars0[self._F,:]

        lindf = np.float32(60.*134/np.sqrt(25.))
        logdf = np.float32(0.01/np.sqrt(25.))
        ff = np.log(logdf*logdf*f0 + logdf*np.sqrt(lindf*lindf + logdf*logdf*f0*f0)) / logdf
        ffmin = np.log(logdf*logdf*self.trueminf + logdf*np.sqrt(lindf*lindf + logdf*logdf*self.trueminf*self.trueminf)) / logdf
        dff = np.random.normal(size=nw).astype(np.float32)
        aboveffmin = ff - ffmin
        oob_flux = (-dff > aboveffmin)
        dff[oob_flux] = -2*aboveffmin[oob_flux] - dff[oob_flux]
        pff = ff + dff
        pf = np.exp(-logdf*pff) * (-lindf*lindf*logdf*logdf+np.exp(2*logdf*pff)) / (2*logdf*logdf)
        # calculate flux distribution prior factor
        dlogf = np.log(pf/f0)
        factor = -self.truealpha*dlogf

        dpos_rms = np.float32(60.*134/np.sqrt(25.))/(np.maximum(f0, pf))
        dx = np.random.normal(size=nw).astype(np.float32)*dpos_rms
        dy = np.random.normal(size=nw).astype(np.float32)*dpos_rms
        starsp[self._X,:] = stars0[self._X,:] + dx
        starsp[self._Y,:] = stars0[self._Y,:] + dy
        starsp[self._F,:] = pf
        self.bounce_off_edges(starsp)

        proposal = Proposal()
        proposal.add_move_stars(idx_move, stars0, starsp)
        proposal.set_factor(factor)
        return proposal

    def birth_death_stars(self):
        lifeordeath = np.random.randint(2)
        nbd = (self.nregx * self.nregy) / 4
        proposal = Proposal()
        # birth
        if lifeordeath and self.n < self.nstar: # need room for at least one source
            nbd = min(nbd, self.nstar-self.n) # add nbd sources, or just as many as will fit
                                    # mildly violates detailed balance when n close to nstar
            # want number of regions in each direction, divided by two, rounded up
            mregx = ((imsz[0] / regsize + 1) + 1) / 2 # assumes that imsz are multiples of regsize
            mregy = ((imsz[1] / regsize + 1) + 1) / 2
            starsb = np.empty((3, nbd), dtype=np.float32)
            starsb[self._X,:] = (np.random.randint(mregx, size=nbd)*2 + self.parity_x + np.random.uniform(size=nbd))*regsize - self.offsetx
            starsb[self._Y,:] = (np.random.randint(mregy, size=nbd)*2 + self.parity_y + np.random.uniform(size=nbd))*regsize - self.offsety
            starsb[self._F,:] = self.trueminf * np.exp(np.random.exponential(scale=1./(self.truealpha-1.),size=nbd))

            # some sources might be generated outside image
            inbounds = self.in_bounds(starsb)
            starsb = starsb.compress(inbounds, axis=1)
            factor = np.full(starsb.shape[1], -self.penalty)

            proposal.add_birth_stars(starsb)
            proposal.set_factor(factor)
        # death
        # does region based death obey detailed balance?
        elif not lifeordeath and self.n > 0: # need something to kill
            idx_reg = self.idx_parity_stars()

            nbd = min(nbd, idx_reg.size) # kill nbd sources, or however many sources remain
            if nbd > 0:
                idx_kill = np.random.choice(idx_reg, size=nbd, replace=False)
                starsk = self.stars.take(idx_kill, axis=1)
                factor = np.full(nbd, self.penalty)

                proposal.add_death_stars(idx_kill, starsk)
                proposal.set_factor(factor)
        return proposal

    def merge_split_stars(self):
        splitsville = np.random.randint(2)
        idx_reg = self.idx_parity_stars()
        sum_f = 0
        low_n = 0
        idx_bright = idx_reg.take(np.flatnonzero(self.stars[self._F, :].take(idx_reg) > 2*self.trueminf)) # in region!
        bright_n = idx_bright.size

        nms = (self.nregx * self.nregy) / 4
        goodmove = False
        proposal = Proposal()
        # split
        if splitsville and self.n > 0 and self.n < self.nstar and bright_n > 0: # need something to split, but don't exceed nstar
            nms = min(nms, bright_n, self.nstar-self.n) # need bright source AND room for split source
            dx = (np.random.normal(size=nms)*self.kickrange).astype(np.float32)
            dy = (np.random.normal(size=nms)*self.kickrange).astype(np.float32)
            idx_move = np.random.choice(idx_bright, size=nms, replace=False)
            stars0 = self.stars.take(idx_move, axis=1)
            x0, y0, f0 = stars0
            fminratio = f0 / self.trueminf
            frac = (1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32)
            
            starsp = np.empty_like(stars0)
            starsp[self._X,:] = x0 + ((1-frac)*dx)
            starsp[self._Y,:] = y0 + ((1-frac)*dy)
            starsp[self._F,:] = f0 * frac
            starsb = np.empty_like(stars0)
            starsb[self._X,:] = x0 - frac*dx
            starsb[self._Y,:] = y0 - frac*dy
            starsb[self._F,:] = f0 * (1-frac)

            # don't want to think about how to bounce split-merge
            # don't need to check if above fmin, because of how frac is decided
            inbounds = np.logical_and(self.in_bounds(starsp), self.in_bounds(starsb))
            stars0 = stars0.compress(inbounds, axis=1)
            starsp = starsp.compress(inbounds, axis=1)
            starsb = starsb.compress(inbounds, axis=1)
            idx_move = idx_move.compress(inbounds)
            fminratio = fminratio.compress(inbounds)
            frac = frac.compress(inbounds)
            nms = idx_move.size
            goodmove = nms > 0
            if goodmove:
                proposal.add_move_stars(idx_move, stars0, starsp)
                proposal.add_birth_stars(starsb)

            # need to calculate factor
            sum_f = stars0[self._F,:]
            invpairs = np.empty(nms)
            for k in xrange(nms):
                xtemp = self.stars[self._X, 0:self.n].copy()
                ytemp = self.stars[self._Y, 0:self.n].copy()
                xtemp[idx_move[k]] = starsp[self._X, k]
                ytemp[idx_move[k]] = starsp[self._Y, k]
                xtemp = np.concatenate([xtemp, starsb[self._X, k:k+1]])
                ytemp = np.concatenate([ytemp, starsb[self._Y, k:k+1]])

                invpairs[k] =  1./neighbours(xtemp, ytemp, self.kickrange, idx_move[k]) #divide by zero
                invpairs[k] += 1./neighbours(xtemp, ytemp, self.kickrange, self.n)
            invpairs *= 0.5
        # merge
        elif not splitsville and idx_reg.size > 1: # need two things to merge!
            nms = min(nms, idx_reg.size/2)
            idx_move = np.empty(nms, dtype=np.int)
            idx_kill = np.empty(nms, dtype=np.int)
            choosable = np.zeros(self.nstar, dtype=np.bool)
            choosable[idx_reg] = True
            nchoosable = float(idx_reg.size)
            invpairs = np.empty(nms)

            for k in xrange(nms):
                idx_move[k] = np.random.choice(self.nstar, p=choosable/nchoosable)
                invpairs[k], idx_kill[k] = neighbours(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], self.kickrange, idx_move[k], generate=True)
                if invpairs[k] > 0:
                    invpairs[k] = 1./invpairs[k]
                # prevent sources from being involved in multiple proposals
                if not choosable[idx_kill[k]]:
                    idx_kill[k] = -1
                if idx_kill[k] != -1:
                    invpairs[k] += 1./neighbours(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], self.kickrange, idx_kill[k])
                    choosable[idx_move[k]] = False
                    choosable[idx_kill[k]] = False
                    nchoosable -= 2
            invpairs *= 0.5

            inbounds = (idx_kill != -1)
            idx_move = idx_move.compress(inbounds)
            idx_kill = idx_kill.compress(inbounds)
            invpairs = invpairs.compress(inbounds)
            nms = idx_move.size
            goodmove = nms > 0

            stars0 = self.stars.take(idx_move, axis=1)
            starsk = self.stars.take(idx_kill, axis=1)

            f0 = stars0[self._F,:]
            fk = starsk[self._F,:]
            sum_f = f0 + fk
            fminratio = sum_f / self.trueminf
            frac = f0 / sum_f
            starsp = np.empty_like(stars0)
            starsp[self._X,:] = frac*stars0[self._X,:] + (1-frac)*starsk[self._X,:]
            starsp[self._Y,:] = frac*stars0[self._Y,:] + (1-frac)*starsk[self._Y,:]
            starsp[self._F,:] = f0 + fk
            if goodmove:
                proposal.add_move_stars(idx_move, stars0, starsp)
                proposal.add_death_stars(idx_kill, starsk)
            # turn bright_n into an array
            bright_n = bright_n - (f0 > 2*self.trueminf) - (fk > 2*self.trueminf) + (starsp[self._F,:] > 2*self.trueminf)
        if goodmove:
            factor = np.log(self.truealpha-1) + (self.truealpha-1)*np.log(self.trueminf) - self.truealpha*np.log(frac*(1-frac)*sum_f) + np.log(2*np.pi*self.kickrange*self.kickrange) - np.log(imsz[0]*imsz[1]) + np.log(1. - 2./fminratio) + np.log(bright_n) + np.log(invpairs) + np.log(sum_f) # last term is Jacobian
            if not splitsville:
                factor *= -1
                factor += self.penalty
            else:
                factor -= self.penalty
            proposal.set_factor(factor)
        return proposal

    def move_galaxies(self):
        idx_move_g = self.idx_parity_galaxies()
        nw = idx_move_g.size
        galaxies0 = self.galaxies.take(idx_move_g, axis=1)
        f0g = galaxies0[self._F,:]

        lindf = np.float32(60.*134/np.sqrt(25.))
        logdf = np.float32(0.01/np.sqrt(25.))
        ff = np.log(logdf*logdf*f0g + logdf*np.sqrt(lindf*lindf + logdf*logdf*f0g*f0g)) / logdf
        ffmin = np.log(logdf*logdf*self.trueminf_g + logdf*np.sqrt(lindf*lindf + logdf*logdf*self.trueminf_g*self.trueminf_g)) / logdf
        dff = np.random.normal(size=nw).astype(np.float32)
        aboveffmin = ff - ffmin
        oob_flux = (-dff > aboveffmin)
        dff[oob_flux] = -2*aboveffmin[oob_flux] - dff[oob_flux]
        pff = ff + dff
        pfg = np.exp(-logdf*pff) * (-lindf*lindf*logdf*logdf+np.exp(2*logdf*pff)) / (2*logdf*logdf)

        dlogfg = np.log(pfg/f0g)
        factor = -self.truealpha_g*dlogfg

        dpos_rms = np.float32(60.*134/np.sqrt(25.))/(np.maximum(f0g, pfg))
        dxg = np.random.normal(size=nw).astype(np.float32)*dpos_rms
        dyg = np.random.normal(size=nw).astype(np.float32)*dpos_rms
        dxxg = np.random.normal(size=nw, scale=0.1).astype(np.float32)
        dxyg = np.random.normal(size=nw, scale=0.1).astype(np.float32)
        dyyg = np.random.normal(size=nw, scale=0.1).astype(np.float32)
        galaxiesp = np.empty_like(galaxies0)
        xx0g, xy0g, yy0g = galaxies0[[self._XX, self._XY, self._YY],:]
        pxxg = xx0g + dxxg
        pxyg = xy0g + dxyg
        pyyg = yy0g + dyyg
        # bounce xx, yy off of zero TODO min radius of galaxies?
        mask = pxxg < 0
        pxxg[mask] *= -1
        mask = pyyg < 0
        pyyg[mask] *= -1
        # put in -inf for illegal ellipses
        mask = (pxxg*pyyg - pxyg*pxyg) <= 0
        pxxg[mask] = xx0g[mask] # make illegal ellipses legal so that prior factor can be calculated, even though not used
        pxyg[mask] = xy0g[mask]
        pyyg[mask] = yy0g[mask]
        factor[mask] = -float('inf')

        galaxiesp[self._X,:] = galaxies0[self._X,:] + dxg
        galaxiesp[self._Y,:] = galaxies0[self._Y,:] + dyg
        galaxiesp[self._F,:] = pfg
        galaxiesp[self._XX,:] = pxxg
        galaxiesp[self._XY,:] = pxyg
        galaxiesp[self._YY,:] = pyyg
        self.bounce_off_edges(galaxiesp)
        # calculate prior factor 
        factor += -self.log_prior_moments(galaxies0) + self.log_prior_moments(galaxiesp)
        
        proposal = Proposal()
        proposal.add_move_galaxies(idx_move_g, galaxies0, galaxiesp)
        proposal.set_factor(factor)
        return proposal

    def birth_death_galaxies(self):
        lifeordeath = np.random.randint(2)
        nbd = (self.nregx * self.nregy) / 4
        proposal = Proposal()
        # birth
        if lifeordeath and self.ng < self.ngalx: # need room for at least one source
            nbd = min(nbd, self.ngalx-self.ng) # add nbd sources, or just as many as will fit
                                    # mildly violates detailed balance when n close to nstar
            # want number of regions in each direction, divided by two, rounded up
            mregx = ((imsz[0] / regsize + 1) + 1) / 2 # assumes that imsz are multiples of regsize
            mregy = ((imsz[1] / regsize + 1) + 1) / 2
            galaxiesb = np.empty((6,nbd))
            galaxiesb[self._X,:] = (np.random.randint(mregx, size=nbd)*2 + self.parity_x + np.random.uniform(size=nbd))*regsize - self.offsetx
            galaxiesb[self._Y,:] = (np.random.randint(mregy, size=nbd)*2 + self.parity_y + np.random.uniform(size=nbd))*regsize - self.offsety
            galaxiesb[self._F,:] = self.trueminf * np.exp(np.random.exponential(scale=1./(self.truealpha-1.),size=nbd))
            galaxiesb[[self._XX, self._XY, self._YY],:] = self.moments_from_prior(self.truermin_g, nbd)

            # some sources might be generated outside image
            inbounds = self.in_bounds(galaxiesb)
            nbd = np.sum(inbounds)
            galaxiesb = galaxiesb.compress(inbounds, axis=1)
            factor = np.full(nbd, -self.penalty_g)

            proposal.add_birth_galaxies(galaxiesb)
            proposal.set_factor(factor)
        # death
        # does region based death obey detailed balance?
        elif not lifeordeath and self.ng > 0: # need something to kill
            idx_reg = self.idx_parity_galaxies()

            nbd = min(nbd, idx_reg.size) # kill nbd sources, or however many sources remain
            if nbd > 0:
                idx_kill_g = np.random.choice(idx_reg, size=nbd, replace=False)
                galaxiesk = self.galaxies.take(idx_kill_g, axis=1)
                factor = np.full(nbd, self.penalty_g)
                proposal.add_death_galaxies(idx_kill_g, galaxiesk)
                proposal.set_factor(factor)
        return proposal

    def star_galaxy(self):
        starorgalx = np.random.randint(2)
        nsg = (self.nregx * self.nregy) / 4
        proposal = Proposal()
        # star -> galaxy
        if starorgalx and self.n > 0 and self.ng < self.ngalx:
            idx_reg = self.idx_parity_stars()
            nsg = min(nsg, min(idx_reg.size, self.ngalx-self.ng))
            if nsg > 0:
                idx_kill = np.random.choice(idx_reg, size=nsg, replace=False)
                starsk = self.stars.take(idx_kill, axis=1)
                galaxiesb = np.empty((6, nsg))
                galaxiesb[[self._X, self._Y, self._F],:] = starsk
                galaxiesb[[self._XX, self._XY, self._YY],:] = self.moments_from_prior(self.truermin_g, nsg)
                factor = np.full(nsg, self.penalty-self.penalty_g) # TODO factor if star and galaxy flux distributions different
                proposal.add_death_stars(idx_kill, starsk)
                proposal.add_birth_galaxies(galaxiesb)
                proposal.set_factor(factor)
        # galaxy -> star
        elif not starorgalx and self.ng > 1 and self.n < self.nstar:
            idx_reg = self.idx_parity_galaxies()
            nsg = min(nsg, min(idx_reg.size, self.nstar-self.n))
            if nsg > 0:
                idx_kill_g = np.random.choice(idx_reg, size=nsg, replace=False)
                galaxiesk = self.galaxies.take(idx_kill_g, axis=1)
                starsb = galaxiesk[[self._X, self._Y, self._F],:].copy()
                factor = np.full(nsg, self.penalty_g-self.penalty) # TODO factor if star and galaxy flux distributions different
                proposal.add_death_galaxies(idx_kill_g, galaxiesk)
                proposal.add_birth_stars(starsb)
                proposal.set_factor(factor)
        return proposal

    def twostars_galaxy(self):
        splitsville = np.random.randint(2)
        idx_reg = self.idx_parity_stars() # stars
        idx_reg_g = self.idx_parity_galaxies() # galaxies
        sum_f = 0
        low_n = 0
        idx_bright = idx_reg_g.take(np.flatnonzero(self.galaxies[self._F, :].take(idx_reg_g) > 2*self.trueminf)) # in region and bright enough to make two stars
        bright_n = idx_bright.size # can only split bright galaxies

        nms = (self.nregx * self.nregy) / 4
        goodmove = False
        proposal = Proposal()
        # split
        if splitsville and self.ng > 0 and self.n < self.nstar-2 and bright_n > 0: # need something to split, but don't exceed nstar
            nms = min(nms, bright_n, (self.nstar-self.n)/2) # need bright galaxy AND room for split stars
            idx_kill_g = np.random.choice(idx_bright, size=nms, replace=False)
            galaxiesk = self.galaxies.take(idx_kill_g, axis=1)
            xkg, ykg, fkg, xxkg, xykg, yykg = galaxiesk
            fminratio = fkg / self.trueminf # again, care about fmin for stars
            frac = (1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32)
            f1Mf = frac * (1. - frac) # frac(1 - frac)
            agalx, theta, phi = from_moments(xxkg, xykg, yykg)
            dx = agalx * np.cos(phi) / np.sqrt(2 * f1Mf)
            dy = agalx * np.sin(phi) / np.sqrt(2 * f1Mf)
            dr2 = dx*dx + dy*dy
            starsb = np.empty((3,nms,2), dtype=np.float32)
            starsb[self._X, :, [0,1]] = xkg + ((1-frac)*dx), xkg - frac*dx
            starsb[self._Y, :, [0,1]] = ykg + ((1-frac)*dy), ykg - frac*dy
            starsb[self._F, :, [0,1]] = fkg * frac         , fkg * (1-frac)

            # don't want to think about how to bounce split-merge
            # don't need to check if above fmin, because of how frac is decided
            inbounds = np.logical_and(self.in_bounds(starsb[:,:,0]), self.in_bounds(starsb[:,:,1]))
            idx_kill_g = idx_kill_g.compress(inbounds)
            galaxiesk = galaxiesk.compress(inbounds, axis=1)
            starsb = starsb.compress(inbounds, axis=1)
            fminratio = fminratio.compress(inbounds)
            frac = frac.compress(inbounds)
            dr2 = dr2.compress(inbounds)
            f1Mf = f1Mf.compress(inbounds)
            theta = theta.compress(inbounds)
            fkg = fkg.compress(inbounds)
            nms = fkg.size
            goodmove = nms > 0
            if goodmove:
                proposal.add_death_galaxies(idx_kill_g, galaxiesk)
                proposal.add_birth_stars(starsb)

            # need star pairs to calculate factor
            sum_f = fkg
            weightoverpairs = np.empty(nms) # w (1/sum w_1 + 1/sum w_2) / 2
            for k in xrange(nms):
                xtemp = self.stars[self._X, 0:self.n].copy()
                ytemp = self.stars[self._Y, 0:self.n].copy()
                xtemp = np.concatenate([xtemp, starsb[0, k:k+1, self._X], starsb[1, k:k+1, self._X]])
                ytemp = np.concatenate([ytemp, starsb[0, k:k+1, self._Y], starsb[1, k:k+1, self._Y]])

                neighi = neighbours(xtemp, ytemp, self.kickrange_g, self.n)
                neighj = neighbours(xtemp, ytemp, self.kickrange_g, self.n+1)
                if neighi > 0 and neighj > 0:
                    weightoverpairs[k] = 1./neighi + 1./neighj
                else:
                    weightoverpairs[k] = 0.
            weightoverpairs *= 0.5 * np.exp(-dr2/(2.*self.kickrange_g*self.kickrange_g))
            weightoverpairs[weightoverpairs == 0] = 1
        # merge
        elif not splitsville and idx_reg.size > 1: # need two things to merge!
            nms = min(nms, idx_reg.size/2, self.ngalx-self.ng)
            idx_kill = np.empty((nms, 2), dtype=np.int)
            choosable = np.zeros(self.nstar, dtype=np.bool)
            choosable[idx_reg] = True
            nchoosable = float(idx_reg.size)
            invpairs = np.empty(nms)

            for k in xrange(nms):
                idx_kill[k,0] = np.random.choice(self.nstar, p=choosable/nchoosable)
                invpairs[k], idx_kill[k,1] = neighbours(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], self.kickrange_g, idx_kill[k,0], generate=True)
                # prevent sources from being involved in multiple proposals
                if not choosable[idx_kill[k,1]]:
                    idx_kill[k,1] = -1
                if idx_kill[k,1] != -1:
                    invpairs[k] = 1./invpairs[k]
                    invpairs[k] += 1./neighbours(self.stars[self._X, 0:self.n], self.stars[self._Y, 0:self.n], self.kickrange_g, idx_kill[k,1])
                    choosable[idx_kill[k,:]] = False
                    nchoosable -= 2
            invpairs *= 0.5

            inbounds = (idx_kill[:,1] != -1)
            idx_kill = idx_kill.compress(inbounds, axis=0)
            invpairs = invpairs.compress(inbounds)
            nms = np.sum(inbounds)
            goodmove = nms > 0

            starsk = self.stars.take(idx_kill, axis=1) # because stars is (3, N) and idx_kill is (nms, 2), this is (3, nms, 2)
            fk = starsk[self._F,:,:]
            dx = starsk[self._X,:,1] - starsk[self._X,:,0]
            dy = starsk[self._X,:,1] - starsk[self._Y,:,0]
            dr2 = dx*dx + dy*dy
            weightoverpairs = np.exp(-dr2/(2.*self.kickrange_g*self.kickrange_g)) * invpairs
            weightoverpairs[weightoverpairs == 0] = 1
            sum_f = np.sum(fk, axis=1)
            fminratio = sum_f / self.trueminf
            frac = fk[:,0] / sum_f
            f1Mf = frac * (1. - frac)
            galaxiesb = np.empty((6, nms))
            galaxiesb[self._X,:] = frac*starsk[self._X,:,0] + (1-frac)*starsk[self._X,:,1]
            galaxiesb[self._Y,:] = frac*starsk[self._Y,:,0] + (1-frac)*starsk[self._Y,:,1]
            galaxiesb[self._F,:] = sum_f
            u = np.random.uniform(low=3e-4, high=1., size=nms).astype(np.float32) #3e-4 for numerics
            theta = np.arccos(u).astype(np.float32)
            galaxiesb[self._XX,:] = f1Mf*(dx*dx+u*u*dy*dy)
            galaxiesb[self._XY,:] = f1Mf*(1-u*u)*dx*dy
            galaxiesb[self._YY,:] = f1Mf*(dy*dy+u*u*dx*dx)
            # this move proposes a splittable galaxy
            bright_n += 1
            if goodmove:
                proposal.add_death_stars(idx_kill, starsk)
                proposal.add_birth_galaxies(galaxiesb)
        if goodmove:
            factor = 2*np.log(self.truealpha-1) - np.log(self.truealpha_g-1) + 2*(self.truealpha-1)*np.log(self.trueminf) - (self.truealpha_g-1)*np.log(self.trueminf_g) - \
                self.truealpha*np.log(f1Mf) - (2*self.truealpha - self.truealpha_g)*np.log(sum_f) - np.log(imsz[0]*imsz[1]) + np.log(1. - 2./fminratio) - \
                np.log(2*np.pi*self.kickrange_g*self.kickrange_g) + np.log(bright_n/(self.ng+1.-splitsville)) + np.log((self.n-1+2*splitsville)*weightoverpairs) + \
                np.log(sum_f) - np.log(4.) - 2*np.log(dr2) - 3*np.log(f1Mf) - np.log(np.cos(theta)) - 3*np.log(np.sin(theta))
            if not splitsville:
                factor *= -1
                factor += 2*self.penalty - self.penalty_g
                factor += self.log_prior_moments(galaxiesb)
            else:
                factor -= 2*self.penalty - self.penalty_g
                factor -= self.log_prior_moments(galaxiesk)
            proposal.set_factor(factor)
        return proposal

    def stargalaxy_galaxy(self):
        splitsville = np.random.randint(2)
        idx_reg_g = self.idx_parity_galaxies()
        sum_f = 0
        low_n = 0
        idx_bright = idx_reg_g.take(np.flatnonzero(self.galaxies[self._F, :].take(idx_reg_g) > self.trueminf + self.trueminf_g)) # in region and bright enough to make s+g
        bright_n = idx_bright.size

        nms = (self.nregx * self.nregy) / 4
        goodmove = False
        proposal = Proposal()
        # split off star
        if splitsville and self.ng > 0 and self.n < self.nstar and bright_n > 0: # need something to split, but don't exceed nstar
            nms = min(nms, bright_n, self.nstar-self.n) # need bright source AND room for split off star
            dx = (np.random.normal(size=nms)*self.kickrange_g).astype(np.float32)
            dy = (np.random.normal(size=nms)*self.kickrange_g).astype(np.float32)
            idx_move_g = np.random.choice(idx_bright, size=nms, replace=False)
            galaxies0 = self.galaxies.take(idx_move_g, axis=1)
            x0g, y0g, f0g, xx0g, xy0g, yy0g = galaxies0
            frac = (self.trueminf_g/f0g + np.random.uniform(size=nms)*(1. - (self.trueminf_g + self.trueminf)/f0g)).astype(np.float32)
            galaxiesp = np.empty_like(galaxies0)
            galaxiesp[self._X,:] = x0g + ((1-frac)*dx)
            galaxiesp[self._Y,:] = y0g + ((1-frac)*dy)
            galaxiesp[self._F,:] = f0g * frac
            pxxg = (xx0g - frac*(1-frac)*dx*dx)/frac
            pxyg = (xy0g - frac*(1-frac)*dx*dy)/frac
            pyyg = (yy0g - frac*(1-frac)*dy*dy)/frac
            galaxiesp[self._XX,:] = pxxg
            galaxiesp[self._XY,:] = pxyg
            galaxiesp[self._YY,:] = pyyg
            starsb = np.empty((3,nms), dtype=np.float32)
            starsb[self._X,:] = x0g - frac*dx
            starsb[self._Y,:] = y0g - frac*dy
            starsb[self._F,:] = f0g * (1-frac)

            # don't want to think about how to bounce split-merge
            # don't need to check if above fmin, because of how frac is decided
            inbounds = np.logical_and(np.logical_and(self.in_bounds(galaxiesp), self.in_bounds(starsb)), \
                    np.logical_and(np.logical_and(pxxg > 0, pyyg > 0), pxxg*pyyg > pxyg*pxyg)) # TODO min galaxy radius, inbounds function for galaxies?
            idx_move_g = idx_move_g.compress(inbounds)
            galaxies0 = galaxies0.compress(inbounds, axis=1)
            galaxiesp = galaxiesp.compress(inbounds, axis=1)
            starsb = starsb.compress(inbounds, axis=1)
            frac = frac.compress(inbounds)
            f0g = f0g.compress(inbounds)
            nms = idx_move_g.size
            goodmove = nms > 0
            if goodmove:
                proposal.add_move_galaxies(idx_move_g, galaxies0, galaxiesp)
                proposal.add_birth_stars(starsb)

            # need to calculate factor
            sum_f = f0g
            invpairs = np.zeros(nms)
            for k in xrange(nms):
                xtemp = self.stars[self._X, 0:self.n].copy()
                ytemp = self.stars[self._Y, 0:self.n].copy()
                xtemp = np.concatenate([xtemp, galaxiesp[self._X, k:k+1], starsb[self._X, k:k+1]])
                ytemp = np.concatenate([ytemp, galaxiesp[self._Y, k:k+1], starsb[self._Y, k:k+1]])

                invpairs[k] =  1./neighbours(xtemp, ytemp, self.kickrange_g, self.n)
        # merge star into galaxy
        elif not splitsville and idx_reg_g.size > 1: # need two things to merge!
            nms = min(nms, idx_reg_g.size)
            idx_move_g = np.random.choice(idx_reg_g, size=nms, replace=False) # choose galaxies and then see if they have neighbours
            idx_kill = np.empty(nms, dtype=np.int)
            choosable = np.full(self.nstar, True, dtype=np.bool)
            nchoosable = float(self.nstar)
            invpairs = np.empty(nms)

            for k in xrange(nms):
                l = idx_move_g[k]
                invpairs[k], idx_kill[k] = neighbours(np.concatenate([self.stars[self._X, 0:self.n], self.galaxies[self._X, l:l+1]]), \
                        np.concatenate([self.stars[self._Y, 0:self.n], self.galaxies[self._Y, l:l+1]]), self.kickrange_g, self.n, generate=True)
                if invpairs[k] > 0:
                    invpairs[k] = 1./invpairs[k]
                else:
                    assert idx_kill[k] == -1
                # prevent sources from being involved in multiple proposals
                if not choosable[idx_kill[k]]:
                    idx_kill[k] = -1
                if idx_kill[k] != -1:
                    choosable[idx_kill[k]] = False

            inbounds = (idx_kill != -1)
            idx_move_g = idx_move_g.compress(inbounds)
            idx_kill = idx_kill.compress(inbounds)
            invpairs = invpairs.compress(inbounds)
            nms = idx_move_g.size
            goodmove = nms > 0

            galaxies0 = self.galaxies.take(idx_move_g, axis=1)
            starsk = self.stars.take(idx_kill, axis=1)
            sum_f = galaxies0[self._F,:] + starsk[self._F,:]
            frac = galaxies0[self._F,:] / sum_f
            dx = galaxies0[self._X,:] - starsk[self._X,:]
            dy = galaxies0[self._Y,:] - starsk[self._Y,:]
            pxxg = frac*galaxies0[self._XX,:] + frac*(1-frac)*dx*dx
            pxyg = frac*galaxies0[self._XY,:] + frac*(1-frac)*dx*dy
            pyyg = frac*galaxies0[self._YY,:] + frac*(1-frac)*dy*dy
            inbounds = (pxxg > 0) * (pyyg > 0) * (pxxg*pyyg > pxyg*pxyg) # make sure ellipse is legal TODO min galaxy radius
            idx_move_g = idx_move_g.compress(inbounds)
            galaxies0 = galaxies0.compress(inbounds, axis=1)
            idx_kill = idx_kill.compress(inbounds)
            invpairs = invpairs.compress(inbounds)
            starsk = starsk.compress(inbounds, axis=1)
            sum_f = sum_f.compress(inbounds)
            frac = frac.compress(inbounds)
            pxxg = pxxg.compress(inbounds)
            pxyg = pxyg.compress(inbounds)
            pyyg = pyyg.compress(inbounds)
            nms = idx_move_g.size
            goodmove = np.logical_and(goodmove, nms > 0)

            if goodmove:
                galaxiesp = np.empty((6, nms))
                galaxiesp[self._X,:] = frac*galaxies0[self._X,:] + (1-frac)*starsk[self._X,:]
                galaxiesp[self._Y,:] = frac*galaxies0[self._Y,:] + (1-frac)*starsk[self._Y,:]
                galaxiesp[self._F,:] = galaxies0[self._F,:] + starsk[self._F,:]
                galaxiesp[self._XX,:] = pxxg
                galaxiesp[self._XY,:] = pxyg
                galaxiesp[self._YY,:] = pyyg
                proposal.add_move_galaxies(idx_move_g, galaxies0, galaxiesp)
                proposal.add_death_stars(idx_kill, starsk)
            # this proposal makes a galaxy that is bright enough to split
            bright_n = bright_n + 1
        if goodmove:
            factor = np.log(self.truealpha-1) - (self.truealpha-1)*np.log(sum_f/self.trueminf) - self.truealpha_g*np.log(frac) - self.truealpha*np.log(1-frac) + \
                    np.log(2*np.pi*self.kickrange_g*self.kickrange_g) - np.log(imsz[0]*imsz[1]) + np.log(1. - (self.trueminf+self.trueminf_g)/sum_f) + \
                    np.log(bright_n/float(self.ng)) + np.log((self.n+1.-splitsville)*invpairs) - 2*np.log(frac)
            if not splitsville:
                factor *= -1
            factor += self.log_prior_moments(galaxiesp) - self.log_prior_moments(galaxies0) # galaxy prior
            if not splitsville:
                factor += self.penalty
            else:
                factor -= self.penalty
            proposal.set_factor(factor)
        return proposal

    def merge_split_galaxies(self):
        splitsville = np.random.randint(2)
        idx_reg = self.idx_parity_galaxies()
        sum_f = 0
        low_n = 0
        idx_bright = idx_reg.take(np.flatnonzero(self.galaxies[self._F, :].take(idx_reg) > 2*self.trueminf_g)) # in region!
        bright_n = idx_bright.size

        nms = (self.nregx * self.nregy) / 4
        goodmove = False
        proposal = Proposal()
        # split
        if splitsville and self.ng > 0 and self.ng < self.ngalx and bright_n > 0: # need something to split, but don't exceed nstar
            nms = min(nms, bright_n, self.ngalx-self.ng) # need bright source AND room for split source
            dx = (np.random.normal(size=nms)*self.kickrange_g).astype(np.float32)
            dy = (np.random.normal(size=nms)*self.kickrange_g).astype(np.float32)
            idx_move_g = np.random.choice(idx_bright, size=nms, replace=False)
            galaxies0 = self.galaxies.take(idx_move_g, axis=1)
            fminratio = galaxies0[self._F,:] / self.trueminf_g
            frac = (1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32)
            frac_xx = np.random.uniform(size=nms).astype(np.float32)
            frac_xy = np.random.uniform(size=nms).astype(np.float32)
            frac_yy = np.random.uniform(size=nms).astype(np.float32)
            xx_p = galaxies0[self._XX,:] - frac*(1-frac)*dx*dx# moments of just galaxy pair
            xy_p = galaxies0[self._XY,:] - frac*(1-frac)*dx*dy
            yy_p = galaxies0[self._YY,:] - frac*(1-frac)*dy*dy
            galaxiesp = np.empty_like(galaxies0)
            galaxiesp[self._X,:] = galaxies0[self._X,:] + ((1-frac)*dx)
            galaxiesp[self._Y,:] = galaxies0[self._Y,:] + ((1-frac)*dy)
            galaxiesp[self._F,:] = galaxies0[self._F,:] * frac
            galaxiesp[self._XX,:] = xx_p * frac_xx / frac
            galaxiesp[self._XY,:] = xy_p * frac_xy / frac
            galaxiesp[self._YY,:] = yy_p * frac_yy / frac
            galaxiesb = np.empty_like(galaxies0)
            galaxiesb[self._X,:] = galaxies0[self._X,:] - frac*dx
            galaxiesb[self._Y,:] = galaxies0[self._Y,:] - frac*dy
            galaxiesb[self._F,:] = galaxies0[self._F,:] * (1-frac)
            galaxiesb[self._XX,:] = xx_p * (1-frac_xx) / (1-frac) # FIXME is this right?
            galaxiesb[self._XY,:] = xy_p * (1-frac_xy) / (1-frac)
            galaxiesb[self._YY,:] = yy_p * (1-frac_yy) / (1-frac)
            # don't want to think about how to bounce split-merge
            # don't need to check if above fmin, because of how frac is decided
            pxxg, pxyg, pyyg = galaxiesp[[self._XX,self._XY,self._YY],:]
            bxxg, bxyg, byyg = galaxiesb[[self._XX,self._XY,self._YY],:]
            inbounds = self.in_bounds(galaxiesp) * \
                       self.in_bounds(galaxiesb) * \
                       (pxxg > 0) * (pyyg > 0) * (pxxg*pyyg > pxyg*pxyg) * \
                       (bxxg > 0) * (byyg > 0) * (bxxg*byyg > bxyg*bxyg) # TODO min galaxy rad
            galaxies0 = galaxies0.compress(inbounds, axis=1)
            galaxiesp = galaxiesp.compress(inbounds, axis=1)
            galaxiesb = galaxiesb.compress(inbounds, axis=1)
            idx_move_g = idx_move_g.compress(inbounds)
            fminratio = fminratio.compress(inbounds)
            frac = frac.compress(inbounds)
            xx_p = xx_p.compress(inbounds)
            xy_p = xy_p.compress(inbounds)
            yy_p = yy_p.compress(inbounds)
            nms = idx_move_g.size
            goodmove = nms > 0

            # need to calculate factor
            sum_f = galaxies0[self._F,:]
            invpairs = np.empty(nms)
            for k in xrange(nms):
                xtemp = self.galaxies[self._X, 0:self.ng].copy()
                ytemp = self.galaxies[self._Y, 0:self.ng].copy()
                xtemp[idx_move_g[k]] = galaxiesp[self._X, k]
                ytemp[idx_move_g[k]] = galaxiesp[self._Y, k]
                xtemp = np.concatenate([xtemp, galaxiesb[self._X, k:k+1]])
                ytemp = np.concatenate([ytemp, galaxiesb[self._Y, k:k+1]])

                invpairs[k] =  1./neighbours(xtemp, ytemp, self.kickrange_g, idx_move_g[k])
                invpairs[k] += 1./neighbours(xtemp, ytemp, self.kickrange_g, self.ng)
            invpairs *= 0.5
        # merge
        elif not splitsville and idx_reg.size > 1: # need two things to merge!
            nms = min(nms, idx_reg.size/2)
            idx_move_g = np.empty(nms, dtype=np.int)
            idx_kill_g = np.empty(nms, dtype=np.int)
            choosable = np.zeros(self.ngalx, dtype=np.bool)
            choosable[idx_reg] = True
            nchoosable = float(idx_reg.size)
            invpairs = np.empty(nms)

            for k in xrange(nms):
                idx_move_g[k] = np.random.choice(self.ngalx, p=choosable/nchoosable)
                invpairs[k], idx_kill_g[k] = neighbours(self.galaxies[self._X, 0:self.ng], self.galaxies[self._Y, 0:self.ng], self.kickrange_g, idx_move_g[k], generate=True)
                if invpairs[k] > 0:
                    invpairs[k] = 1./invpairs[k]
                # prevent sources from being involved in multiple proposals
                if not choosable[idx_kill_g[k]]:
                    idx_kill_g[k] = -1
                if idx_kill_g[k] != -1:
                    invpairs[k] += 1./neighbours(self.galaxies[self._X, 0:self.ng], self.galaxies[self._Y, 0:self.ng], self.kickrange_g, idx_kill_g[k])
                    choosable[idx_move_g[k]] = False
                    choosable[idx_kill_g[k]] = False
                    nchoosable -= 2
            invpairs *= 0.5

            inbounds = (idx_kill_g != -1)
            idx_move_g = idx_move_g.compress(inbounds)
            idx_kill_g = idx_kill_g.compress(inbounds)
            invpairs = invpairs.compress(inbounds)

            galaxies0 = self.galaxies.take(idx_move_g, axis=1)
            galaxiesk = self.galaxies.take(idx_kill_g, axis=1)
            sum_f = galaxies0[self._F,:] + galaxiesk[self._F,:]
            fminratio = sum_f / self.trueminf_g
            frac = galaxies0[self._F,:] / sum_f

            galaxiesp = np.empty_like(galaxies0)
            galaxiesp[self._X,:] = frac*galaxies0[self._X,:] + (1-frac)*galaxiesk[self._X,:]
            galaxiesp[self._Y,:] = frac*galaxies0[self._Y,:] + (1-frac)*galaxiesk[self._Y,:]
            galaxiesp[self._F,:] = galaxies0[self._F,:] + galaxiesk[self._F,:]
            dx = galaxies0[self._X,:] - galaxiesk[self._X,:]
            dy = galaxies0[self._Y,:] - galaxiesk[self._Y,:]
            xx_p = frac*galaxies0[self._XX,:] + (1-frac)*galaxiesk[self._XX,:]
            xy_p = frac*galaxies0[self._XY,:] + (1-frac)*galaxiesk[self._XY,:]
            yy_p = frac*galaxies0[self._YY,:] + (1-frac)*galaxiesk[self._YY,:]
            galaxiesp[self._XX,:] = xx_p + frac*(1-frac)*dx*dx
            galaxiesp[self._XY,:] = xy_p + frac*(1-frac)*dx*dy
            galaxiesp[self._YY,:] = yy_p + frac*(1-frac)*dy*dy

            pxxg, pxyg, pyyg = galaxiesp[[self._XX,self._XY,self._YY],:]
            inbounds = ((pxxg > 0) * (pyyg > 0) * (pxxg*pyyg > pxyg*pxyg)) # ellipse legal TODO minimum radius
            galaxies0 = galaxies0.compress(inbounds, axis=1)
            galaxiesk = galaxiesk.compress(inbounds, axis=1)
            galaxiesp = galaxiesp.compress(inbounds, axis=1)
            idx_move_g = idx_move_g.compress(inbounds)
            idx_kill_g = idx_kill_g.compress(inbounds)
            invpairs = invpairs.compress(inbounds)
            sum_f = sum_f.compress(inbounds)
            fminratio = fminratio.compress(inbounds)
            frac = frac.compress(inbounds)
            xx_p = xx_p.compress(inbounds)
            xy_p = xy_p.compress(inbounds)
            yy_p = yy_p.compress(inbounds)

            
            nms = idx_move_g.size
            goodmove = nms > 0
            if goodmove:
                proposal.add_move_galaxies(idx_move_g, galaxies0, galaxiesp)
                proposal.add_death_galaxies(idx_kill_g, galaxiesk)

            # turn bright_n into an array
            bright_n = bright_n - (galaxies0[self._F,:] > 2*self.trueminf_g) - (galaxiesk[self._F,:] > 2*self.trueminf_g) + (galaxiesp[self._F,:] > 2*self.trueminf_g)
        if goodmove:
            factor = np.log(self.truealpha_g-1) + (self.truealpha_g-1)*np.log(self.trueminf) - self.truealpha_g*np.log(frac*(1-frac)*sum_f) + \
                np.log(2*np.pi*self.kickrange_g*self.kickrange_g) - np.log(imsz[0]*imsz[1]) + np.log(1. - 2./fminratio) + np.log(bright_n) + np.log(invpairs) + \
                np.log(sum_f) + np.log(xx_p) + np.log(np.abs(xy_p)) + np.log(yy_p) - 3*np.log(frac) - 3*np.log(1-frac) # last line is Jacobian
            if not splitsville:
                factor *= -1
                factor += self.penalty_g
                factor += self.log_prior_moments(galaxiesp) - self.log_prior_moments(galaxies0) - self.log_prior_moments(galaxiesk)
            else:
                factor -= self.penalty_g
                factor += self.log_prior_moments(galaxiesp) - self.log_prior_moments(galaxies0) + self.log_prior_moments(galaxiesb)
            proposal.set_factor(factor)
        return proposal

nsamp = 1000
nloop = 1000
nstar = Model.nstar
nsample = np.zeros(nsamp, dtype=np.int32)
xsample = np.zeros((nsamp, nstar), dtype=np.float32)
ysample = np.zeros((nsamp, nstar), dtype=np.float32)
fsample = np.zeros((nsamp, nstar), dtype=np.float32)
ngsample = np.zeros(nsamp, dtype=np.int32)
xgsample = np.zeros((nsamp, nstar), dtype=np.float32)
ygsample = np.zeros((nsamp, nstar), dtype=np.float32)
fgsample = np.zeros((nsamp, nstar), dtype=np.float32)
xxgsample = np.zeros((nsamp, nstar), dtype=np.float32)
xygsample = np.zeros((nsamp, nstar), dtype=np.float32)
yygsample = np.zeros((nsamp, nstar), dtype=np.float32)

models = [Model() for k in xrange(ntemps)]

if visual:
    plt.ion()
    plt.figure(figsize=(15,5))
for j in xrange(nsamp):
    chi2_all = np.zeros(ntemps)
    print 'Loop', j

    temptemp = max(50 - 0.1*j, 1)
    for k in xrange(ntemps):
        _, _, chi2_all[k] = models[k].run_sampler(temptemp, visual=(k==0)*visual)

    for k in xrange(ntemps-1, 0, -1):
        logfac = (chi2_all[k-1] - chi2_all[k]) * (1./temps[k-1] - 1./temps[k]) / 2.
        if np.log(np.random.uniform()) < logfac:
            print 'swapped', k-1, k
            models[k-1], models[k] = models[k], models[k-1]

    nsample[j] = models[0].n
    xsample[j,:] = models[0].stars[Model._X, :]
    ysample[j,:] = models[0].stars[Model._Y, :]
    fsample[j,:] = models[0].stars[Model._F, :]
    ngsample[j] = models[0].ng
    xgsample[j,:] = models[0].galaxies[Model._X, :]
    ygsample[j,:] = models[0].galaxies[Model._Y, :]
    fgsample[j,:] = models[0].galaxies[Model._F, :]
    xxgsample[j,:] = models[0].galaxies[Model._XX, :]
    xygsample[j,:] = models[0].galaxies[Model._XY, :]
    yygsample[j,:] = models[0].galaxies[Model._YY, :]

print 'saving...'
np.savez('chain.npz', n=nsample, x=xsample, y=ysample, f=fsample, ng=ngsample, xg=xgsample, yg=ygsample, fg=fgsample, xxg=xxgsample, xyg=xygsample, yyg=yygsample)
