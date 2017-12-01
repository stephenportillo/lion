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

class Model:
    # should these be class or instance variables?
    nstar = 100
    trueminf = np.float32(250.*136)
    truealpha = np.float32(2.00)
    back = trueback

    ngalx = 100
    trueminf_g = np.float32(250.*136)
    truealpha_g = np.float32(2.00)

    gridphon, amplphon = retr_sers(sersindx=2.)

    penalty = 1.5
    penalty_g = 3.0
    kickrange = 1.
    kickrange_g = 1.

    def __init__(self):
        self.n = np.random.randint(self.nstar)+1
        self.x = (np.random.uniform(size=self.nstar)*(imsz[0]-1)).astype(np.float32)
        self.y = (np.random.uniform(size=self.nstar)*(imsz[1]-1)).astype(np.float32)
        self.f = self.trueminf * np.exp(np.random.exponential(scale=1./(self.truealpha-1.),size=self.nstar)).astype(np.float32)
        self.x[self.n:] = 0.
        self.y[self.n:] = 0.
        self.f[self.n:] = 0.
        self.ng = np.random.randint(self.ngalx)+1
        self.xg = (np.random.uniform(size=self.ngalx)*(imsz[0]-1)).astype(np.float32)
        self.yg = (np.random.uniform(size=self.ngalx)*(imsz[1]-1)).astype(np.float32)
        self.fg = self.trueminf_g * np.exp(np.random.exponential(scale=1./(self.truealpha_g-1.),size=self.ngalx)).astype(np.float32)
        self.truermin_g = np.float32(1.00)
        self.xxg, self.xyg, self.yyg = self.moments_from_prior(self.truermin_g, self.ngalx)
        self.xg[self.ng:] = 0
        self.yg[self.ng:] = 0
        self.fg[self.ng:] = 0
        self.xxg[self.ng:]= 0
        self.xyg[self.ng:]= 0
        self.yyg[self.ng:]= 0

    # should offsetx/y, parity_x/y be instance variables?

    def moments_from_prior(self, truermin_g, ngalx, slope=np.float32(4)):
        rg = truermin_g*np.exp(np.random.exponential(scale=1./(slope-1.),size=ngalx)).astype(np.float32)
        ug = np.random.uniform(low=3e-4, high=1., size=ngalx).astype(np.float32) #3e-4 for numerics
        thetag = np.arccos(ug).astype(np.float32)
        phig = (np.random.uniform(size=ngalx)*np.pi - np.pi/2.).astype(np.float32)
        return to_moments(rg, thetag, phig)

    def log_prior_moments(self, xx, xy, yy):
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
            evalx = self.x[0:self.n]
            evaly = self.y[0:self.n]
            evalf = self.f[0:self.n]
        else:
            xposphon, yposphon, specphon = retr_tranphon(self.gridphon, self.amplphon, self.xg[0:self.ng], self.yg[0:self.ng], self.fg[0:self.ng], \
                self.xxg[0:self.ng], self.xyg[0:self.ng], self.yyg[0:self.ng])
            evalx = np.concatenate([self.x[0:self.n], xposphon]).astype(np.float32)
            evaly = np.concatenate([self.y[0:self.n], yposphon]).astype(np.float32)
            evalf = np.concatenate([self.f[0:self.n], specphon]).astype(np.float32)
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
            factor = None # best way to incorporate acceptance ratio factors?
            goodmove = False

            # should regions be perturbed randomly or systematically?
            self.parity_x = np.random.randint(2)
            self.parity_y = np.random.randint(2)

            idx_move = None
            idx_move_g = None
            do_birth = False
            do_birth_g = False
            idx_kill = None
            idx_kill_g = None

            movetypes = ['MOVE *', 'BD *', 'MS *', 'MOVE g', 'BD g', '*-g', '**-g', '*g-g', 'MS g']

            if rtype == 0:
                idx_move, x0, y0, f0, px, py, pf, goodmove, factor = self.move_stars()
            elif rtype == 1:
                do_birth, bx, by, bf, idx_kill, xk, yk, fk, goodmove, factor = self.birth_death_stars()
            elif rtype == 2:
                idx_move, x0, y0, f0, px, py, pf, do_birth, bx, by, bf, idx_kill, xk, yk, fk, goodmove, factor = self.merge_split_stars()
            elif rtype == 3:
                idx_move_g, x0g, y0g, f0g, xx0g, xy0g, yy0g, pxg, pyg, pfg, pxxg, pxyg, pyyg, goodmove, factor = self.move_galaxies()
            elif rtype == 4:
                do_birth_g, bxg, byg, bfg, bxxg, bxyg, byyg, idx_kill_g, xkg, ykg, fkg, xxkg, xykg, yykg, goodmove, factor = self.birth_death_galaxies()
            elif rtype == 5:
                do_birth, bx, by, bf, do_birth_g, bxg, byg, bfg, bxxg, bxyg, byyg, idx_kill, xk, yk, fk, idx_kill_g, xkg, ykg, fkg, xxkg, xykg, yykg, goodmove, factor = self.star_galaxy()
            elif rtype == 6:
                do_birth, bx, by, bf, do_birth_g, bxg, byg, bfg, bxxg, bxyg, byyg, idx_kill, xk, yk, fk, idx_kill_g, xkg, ykg, fkg, xxkg, xykg, yykg, goodmove, factor = self.twostars_galaxy()
            elif rtype == 7:
                idx_move_g, x0g, y0g, f0g, xx0g, xy0g, yy0g, pxg, pyg, pfg, pxxg, pxyg, pyyg, do_birth, bx, by, bf, idx_kill, xk, yk, fk, goodmove, factor = self.stargalaxy_galaxy()
            elif rtype == 8:
                idx_move_g, x0g, y0g, f0g, xx0g, xy0g, yy0g, pxg, pyg, pfg, pxxg, pxyg, pyyg, do_birth_g, bxg, byg, bfg, bxxg, bxyg, byyg, idx_kill_g, xkg, ykg, fkg, xxkg, xykg, yykg, goodmove, factor = self.merge_split_galaxies()

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
                if idx_move_g is not None:
                    xposphon, yposphon, fluxphon = retr_tranphon(self.gridphon, self.amplphon, np.concatenate([x0g, pxg]), np.concatenate([y0g, pyg]), np.concatenate([-f0g, pfg]), \
                        np.concatenate([xx0g, pxxg]), np.concatenate([xy0g, pxyg]), np.concatenate([yy0g, pyyg]))
                    xtemp.append(xposphon)
                    ytemp.append(yposphon)
                    ftemp.append(fluxphon)
                if do_birth:
                    xtemp.append(bx.flatten())
                    ytemp.append(by.flatten())
                    ftemp.append(bf.flatten())
                if idx_kill is not None:
                    xtemp.append(xk.flatten())
                    ytemp.append(yk.flatten())
                    ftemp.append(-fk.flatten())
                if do_birth_g:
                    xposphon, yposphon, fluxphon = retr_tranphon(self.gridphon, self.amplphon, bxg, byg, bfg, bxxg, bxyg, byyg)
                    xtemp.append(xposphon)
                    ytemp.append(yposphon)
                    ftemp.append(fluxphon)
                if idx_kill_g is not None:
                    xposphon, yposphon, fluxphon = retr_tranphon(self.gridphon, self.amplphon, xkg, ykg, -fkg, xxkg, xykg, yykg)
                    xtemp.append(xposphon)
                    ytemp.append(yposphon)
                    ftemp.append(fluxphon)

                dmodel, diff2 = image_model_eval(np.concatenate(xtemp), np.concatenate(ytemp), np.concatenate(ftemp), dback, imsz, nc, cf, weights=weight, ref=resid, lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety)
                plogL = -0.5*diff2
                dt2[i] = time.clock() - t2

                t3 = time.clock()
                refx = None
                refy = None
                if idx_move is not None:
                    refx = x0
                    refy = y0
                else: # merges and splits evaluated in idx_move region
                    if idx_move_g is not None:
                        refx = x0g
                        refy = y0g
                    elif do_birth:
                        refx = bx if bx.ndim == 1 else bx[:,0]
                        refy = by if by.ndim == 1 else by[:,0]
                    elif idx_kill is not None:
                        refx = xk if xk.ndim == 1 else xk[:,0]
                        refy = yk if yk.ndim == 1 else yk[:,0]
                    elif do_birth_g:
                        refx = bxg
                        refy = byg
                    elif idx_kill_g is not None:
                        refx = xkg
                        refy = ykg
                regionx = get_region(refx, self.offsetx, regsize)
                regiony = get_region(refy, self.offsety, regsize)

                plogL[(1-self.parity_y)::2,:] = float('-inf') # don't accept off-parity regions
                plogL[:,(1-self.parity_x)::2] = float('-inf')
                dlogP = (plogL - logL) / temperature
                if factor is not None:
                    dlogP[regiony, regionx] += factor
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
                if idx_move is not None:
                    px_a = px.compress(acceptprop)
                    py_a = py.compress(acceptprop)
                    pf_a = pf.compress(acceptprop)
                    idx_move_a = idx_move.compress(acceptprop)
                    self.x[idx_move_a] = px_a
                    self.y[idx_move_a] = py_a
                    self.f[idx_move_a] = pf_a
                if idx_move_g is not None:
                    pxg_a = pxg.compress(acceptprop)
                    pyg_a = pyg.compress(acceptprop)
                    pfg_a = pfg.compress(acceptprop)
                    pxxg_a=pxxg.compress(acceptprop)
                    pxyg_a=pxyg.compress(acceptprop)
                    pyyg_a=pyyg.compress(acceptprop)
                    idx_move_a = idx_move_g.compress(acceptprop)
                    self.xg[idx_move_a] = pxg_a
                    self.yg[idx_move_a] = pyg_a
                    self.fg[idx_move_a] = pfg_a
                    self.xxg[idx_move_a]=pxxg_a
                    self.xyg[idx_move_a]=pxyg_a
                    self.yyg[idx_move_a]=pyyg_a
                if do_birth:
                    bx_a = bx.compress(acceptprop, axis=0).flatten()
                    by_a = by.compress(acceptprop, axis=0).flatten()
                    bf_a = bf.compress(acceptprop, axis=0).flatten()
                    num_born = bf_a.size # works for 1D or 2D
                    self.x[self.n:self.n+num_born] = bx_a
                    self.y[self.n:self.n+num_born] = by_a
                    self.f[self.n:self.n+num_born] = bf_a
                    self.n += num_born
                if do_birth_g:
                    bxg_a = bxg.compress(acceptprop)
                    byg_a = byg.compress(acceptprop)
                    bfg_a = bfg.compress(acceptprop)
                    bxxg_a=bxxg.compress(acceptprop)
                    bxyg_a=bxyg.compress(acceptprop)
                    byyg_a=byyg.compress(acceptprop)
                    num_born = np.count_nonzero(acceptprop)
                    self.xg[self.ng:self.ng+num_born] = bxg_a
                    self.yg[self.ng:self.ng+num_born] = byg_a
                    self.fg[self.ng:self.ng+num_born] = bfg_a
                    self.xxg[self.ng:self.ng+num_born]=bxxg_a
                    self.xyg[self.ng:self.ng+num_born]=bxyg_a
                    self.yyg[self.ng:self.ng+num_born]=byyg_a
                    self.ng += num_born
                if idx_kill is not None:
                    idx_kill_a = idx_kill.compress(acceptprop, axis=0).flatten()
                    num_kill = idx_kill_a.size
                    # nstar is correct, not n, because x,y,f are full nstar arrays
                    self.x[0:self.nstar-num_kill] = np.delete(self.x, idx_kill_a)
                    self.y[0:self.nstar-num_kill] = np.delete(self.y, idx_kill_a)
                    self.f[0:self.nstar-num_kill] = np.delete(self.f, idx_kill_a)
                    self.x[self.nstar-num_kill:] = 0
                    self.y[self.nstar-num_kill:] = 0
                    self.f[self.nstar-num_kill:] = 0
                    self.n -= num_kill
                if idx_kill_g is not None:
                    idx_kill_a = idx_kill_g.compress(acceptprop)
                    num_kill = idx_kill_a.size
                    # like above, ngalx is correct
                    self.xg[0:self.ngalx-num_kill] = np.delete(self.xg, idx_kill_a)
                    self.yg[0:self.ngalx-num_kill] = np.delete(self.yg, idx_kill_a)
                    self.fg[0:self.ngalx-num_kill] = np.delete(self.fg, idx_kill_a)
                    self.xxg[0:self.ngalx-num_kill]= np.delete(self.xxg, idx_kill_a)
                    self.xyg[0:self.ngalx-num_kill]= np.delete(self.xyg, idx_kill_a)
                    self.yyg[0:self.ngalx-num_kill]= np.delete(self.yyg, idx_kill_a)
                    self.xg[self.ngalx-num_kill:] = 0
                    self.yg[self.ngalx-num_kill:] = 0
                    self.fg[self.ngalx-num_kill:] = 0
                    self.xxg[self.ngalx-num_kill:]= 0
                    self.xyg[self.ngalx-num_kill:]= 0
                    self.yyg[self.ngalx-num_kill:]= 0
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
        '''print 'Acceptance'+fmtstr % (np.mean(accept), np.mean(accept[movetype == 0]), np.mean(accept[movetype == 2]), np.mean(accept[movetype == 3]), np.mean(accept[movetype==4]), np.mean(accept[movetype==5]), np.mean(accept[movetype==6]), np.mean(accept[movetype==8]), np.mean(accept[movetype==9]))
        print 'Out of bounds'+fmtstr % (np.mean(outbounds), np.mean(outbounds[movetype == 0]), np.mean(outbounds[movetype == 2]), np.mean(outbounds[movetype == 3]), np.mean(outbounds[movetype==4]), np.mean(outbounds[movetype==5]), np.mean(outbounds[movetype==6]), np.mean(outbounds[movetype==8]), np.mean(outbounds[movetype==9]))
        print '# src pert\t(all) %0.1f (P) %0.1f (B-D) %0.1f (M-S) %0.1f' % (np.mean(nmov), np.mean(nmov[movetype == 0]), np.mean(nmov[movetype == 2]), np.mean(nmov[movetype == 3]))
        print '-'*16
        dt1 *= 1000
        dt2 *= 1000
        dt3 *= 1000
        print 'Proposal (ms)\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(dt1), np.mean(dt1[movetype == 0]) , np.mean(dt1[movetype == 2]), np.mean(dt1[movetype == 3]))
        print 'Likelihood (ms)\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(dt2), np.mean(dt2[movetype == 0]) , np.mean(dt2[movetype == 2]), np.mean(dt2[movetype == 3]))
        print 'Implement (ms)\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f' % (np.mean(dt3), np.mean(dt3[movetype == 0]) , np.mean(dt3[movetype == 2]), np.mean(dt3[movetype == 3]))
        print '='*16'''

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
                plt.scatter(self.x[0:self.n], self.y[0:self.n], marker='x', s=self.f[0:self.n]/sizefac, color='r')
                if strgmode == 'galx':
                    plt.scatter(self.xg[0:self.ng], self.yg[0:self.ng], marker='2', s=self.fg[0:self.ng]/sizefac, color='r')
                    a, theta, phi = from_moments(self.xxg[0:self.ng], self.xyg[0:self.ng], self.yyg[0:self.ng])
                    plt.scatter(self.xg[0:self.ng], self.yg[0:self.ng], marker='o', s=4*a*a, edgecolors='red', facecolors='none')
                plt.xlim(-0.5, imsz[0]-0.5)
                plt.ylim(-0.5, imsz[1]-0.5)
                plt.subplot(1,3,2)
                plt.imshow(resid*np.sqrt(weight), origin='lower', interpolation='none', cmap='bwr', vmin=-5, vmax=5)
                if j == 0:
                    plt.tight_layout()
                plt.subplot(1,3,3)

                if datatype == 'mock':
                    plt.hist(np.log10(truef), range=(np.log10(self.trueminf), np.log10(np.max(truef))), log=True, alpha=0.5, label=labldata, histtype='step')
                    plt.hist(np.log10(self.f[0:self.n]), range=(np.log10(self.trueminf), np.log10(np.max(truef))), log=True, alpha=0.5, label='Chain', histtype='step')
                else:
                    plt.hist(np.log10(self.f[0:self.n]), range=(np.log10(self.trueminf), np.ceil(np.log10(np.max(self.f[0:self.n])))), log=True, alpha=0.5, label='Chain', histtype='step')
                plt.legend()
                plt.xlabel('log10 flux')
                plt.ylim((0.5, self.nstar))
                plt.draw()
                plt.pause(1e-5)


        return self.n, self.ng, chi2

    def move_stars(self): 
        idx_move = idx_parity(self.x, self.y, self.n, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize)
        nw = idx_move.size
        f0 = self.f.take(idx_move)

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
        x0 = self.x.take(idx_move)
        y0 = self.y.take(idx_move)
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
        return idx_move, x0, y0, f0, px, py, pf, goodmove, factor

    def birth_death_stars(self):
        lifeordeath = np.random.randint(2)
        nbd = (self.nregx * self.nregy) / 4
        # birth
        if lifeordeath and self.n < self.nstar: # need room for at least one source
            nbd = min(nbd, self.nstar-self.n) # add nbd sources, or just as many as will fit
                                    # mildly violates detailed balance when n close to nstar
            # want number of regions in each direction, divided by two, rounded up
            mregx = ((imsz[0] / regsize + 1) + 1) / 2 # assumes that imsz are multiples of regsize
            mregy = ((imsz[1] / regsize + 1) + 1) / 2
            bx = ((np.random.randint(mregx, size=nbd)*2 + self.parity_x + np.random.uniform(size=nbd))*regsize - self.offsetx).astype(np.float32)
            by = ((np.random.randint(mregy, size=nbd)*2 + self.parity_y + np.random.uniform(size=nbd))*regsize - self.offsety).astype(np.float32)
            bf = self.trueminf * np.exp(np.random.exponential(scale=1./(self.truealpha-1.),size=nbd)).astype(np.float32)

            # some sources might be generated outside image
            inbounds = (bx > 0) * (bx < (imsz[0] -1)) * (by > 0) * (by < imsz[1] - 1)
            idx_in = np.flatnonzero(inbounds)
            bx = bx.take(idx_in)
            by = by.take(idx_in)
            bf = bf.take(idx_in)
            do_birth = True
            factor = np.full(idx_in.size, -self.penalty)
            goodmove = True
            return do_birth, bx, by, bf, None, None, None, None, goodmove, factor
        # death
        # does region based death obey detailed balance?
        elif not lifeordeath and self.n > 0: # need something to kill
            idx_reg = idx_parity(self.x, self.y, self.n, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize)

            nbd = min(nbd, idx_reg.size) # kill nbd sources, or however many sources remain
            if nbd > 0:
                idx_kill = np.random.choice(idx_reg, size=nbd, replace=False)
                xk = self.x.take(idx_kill)
                yk = self.y.take(idx_kill)
                fk = self.f.take(idx_kill)
                factor = np.full(nbd, self.penalty)
                goodmove = True
                return False, None, None, None, idx_kill, xk, yk, fk, goodmove, factor
            else:
                goodmove = False
                return False, None, None, None, None, None, None, None, False, 0

    def merge_split_stars(self):
        splitsville = np.random.randint(2)
        idx_reg = idx_parity(self.x, self.y, self.n, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize)
        sum_f = 0
        low_n = 0
        idx_bright = idx_reg.take(np.flatnonzero(self.f.take(idx_reg) > 2*self.trueminf)) # in region!
        bright_n = idx_bright.size

        nms = (self.nregx * self.nregy) / 4
        goodmove = False
        # split
        if splitsville and self.n > 0 and self.n < self.nstar and bright_n > 0: # need something to split, but don't exceed nstar
            nms = min(nms, bright_n, self.nstar-self.n) # need bright source AND room for split source
            dx = (np.random.normal(size=nms)*self.kickrange).astype(np.float32)
            dy = (np.random.normal(size=nms)*self.kickrange).astype(np.float32)
            idx_move = np.random.choice(idx_bright, size=nms, replace=False)
            x0 = self.x.take(idx_move)
            y0 = self.y.take(idx_move)
            f0 = self.f.take(idx_move)
            fminratio = f0 / self.trueminf
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
            invpairs = np.zeros(nms)
            for k in xrange(nms):
                xtemp = self.x[0:self.n].copy()
                ytemp = self.y[0:self.n].copy()
                xtemp[idx_move[k]] = px[k]
                ytemp[idx_move[k]] = py[k]
                xtemp = np.concatenate([xtemp, bx[k:k+1]])
                ytemp = np.concatenate([ytemp, by[k:k+1]])

                invpairs[k] =  1./neighbours(xtemp, ytemp, self.kickrange, idx_move[k])
                invpairs[k] += 1./neighbours(xtemp, ytemp, self.kickrange, self.n)
            invpairs *= 0.5
        # merge
        elif not splitsville and idx_reg.size > 1: # need two things to merge!
            nms = min(nms, idx_reg.size/2)
            idx_move = np.zeros(nms, dtype=np.int)
            idx_kill = np.zeros(nms, dtype=np.int)
            choosable = np.zeros(self.nstar, dtype=np.bool)
            choosable[idx_reg] = True
            nchoosable = float(np.count_nonzero(choosable))
            invpairs = np.zeros(nms)

            for k in xrange(nms):
                idx_move[k] = np.random.choice(self.nstar, p=choosable/nchoosable)
                invpairs[k], idx_kill[k] = neighbours(self.x[0:self.n], self.y[0:self.n], self.kickrange, idx_move[k], generate=True)
                if invpairs[k] > 0:
                    invpairs[k] = 1./invpairs[k]
                # prevent sources from being involved in multiple proposals
                if not choosable[idx_kill[k]]:
                    idx_kill[k] = -1
                if idx_kill[k] != -1:
                    invpairs[k] += 1./neighbours(self.x[0:self.n], self.y[0:self.n], self.kickrange, idx_kill[k])
                    choosable[idx_move[k]] = False
                    choosable[idx_kill[k]] = False
                    nchoosable -= 2
            invpairs *= 0.5

            inbounds = (idx_kill != -1)
            idx_in = np.flatnonzero(inbounds)
            nms = idx_in.size
            nw = nms
            idx_move = idx_move.take(idx_in)
            idx_kill = idx_kill.take(idx_in)
            invpairs = invpairs.take(idx_in)
            goodmove = idx_in.size > 0

            x0 = self.x.take(idx_move)
            y0 = self.y.take(idx_move)
            f0 = self.f.take(idx_move)
            xk = self.x.take(idx_kill)
            yk = self.y.take(idx_kill)
            fk = self.f.take(idx_kill)
            sum_f = f0 + fk
            fminratio = sum_f / self.trueminf
            frac = f0 / sum_f
            px = frac*x0 + (1-frac)*xk
            py = frac*y0 + (1-frac)*yk
            pf = f0 + fk
            # turn bright_n into an array
            bright_n = bright_n - (f0 > 2*self.trueminf) - (fk > 2*self.trueminf) + (pf > 2*self.trueminf)
        if goodmove:
            factor = np.log(self.truealpha-1) + (self.truealpha-1)*np.log(self.trueminf) - self.truealpha*np.log(frac*(1-frac)*sum_f) + np.log(2*np.pi*self.kickrange*self.kickrange) - np.log(imsz[0]*imsz[1]) + np.log(1. - 2./fminratio) + np.log(bright_n) + np.log(invpairs) + np.log(sum_f) # last term is Jacobian
            if not splitsville:
                factor *= -1
                factor += self.penalty
                return idx_move, x0, y0, f0, px, py, pf, False, None, None, None, idx_kill, xk, yk, fk, goodmove, factor
            else:
                factor -= self.penalty
                return idx_move, x0, y0, f0, px, py, pf, True, bx, by, bf, None, None, None, None, goodmove, factor
        else:
            return None, None, None, None, None, None, None, False, None, None, None, None, None, None, None, goodmove, 0

    def move_galaxies(self):
        idx_move_g = idx_parity(self.xg, self.yg, self.ng, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize)
        nw = idx_move_g.size
        f0g = self.fg.take(idx_move_g)

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
        x0g = self.xg.take(idx_move_g)
        y0g = self.yg.take(idx_move_g)
        xx0g= self.xxg.take(idx_move_g)
        xy0g= self.xyg.take(idx_move_g)
        yy0g= self.yyg.take(idx_move_g)
        pxg = x0g + dxg
        pyg = y0g + dyg
        pxxg=xx0g + dxxg
        pxyg=xy0g + dxyg
        pyyg=yy0g + dyyg
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

        # calculate prior factor 
        factor += -self.log_prior_moments(xx0g, xy0g, yy0g) + self.log_prior_moments(pxxg, pxyg, pyyg)
        
        # bounce off of edges of image
        mask = pxg < 0
        pxg[mask] *= -1
        mask = pxg > (imsz[0] - 1)
        pxg[mask] *= -1
        pxg[mask] += 2*(imsz[0] - 1)
        mask = pyg < 0
        pyg[mask] *= -1
        mask = pyg > (imsz[1] - 1)
        pyg[mask] *= -1
        pyg[mask] += 2*(imsz[1] - 1)
        goodmove = True

        return idx_move_g, x0g, y0g, f0g, xx0g, xy0g, yy0g, pxg, pyg, pfg, pxxg, pxyg, pyyg, goodmove, factor

    def birth_death_galaxies(self):
        lifeordeath = np.random.randint(2)
        nbd = (self.nregx * self.nregy) / 4
        # birth
        if lifeordeath and self.ng < self.ngalx: # need room for at least one source
            nbd = min(nbd, self.ngalx-self.ng) # add nbd sources, or just as many as will fit
                                    # mildly violates detailed balance when n close to nstar
            # want number of regions in each direction, divided by two, rounded up
            mregx = ((imsz[0] / regsize + 1) + 1) / 2 # assumes that imsz are multiples of regsize
            mregy = ((imsz[1] / regsize + 1) + 1) / 2
            bxg = ((np.random.randint(mregx, size=nbd)*2 + self.parity_x + np.random.uniform(size=nbd))*regsize - self.offsetx).astype(np.float32)
            byg = ((np.random.randint(mregy, size=nbd)*2 + self.parity_y + np.random.uniform(size=nbd))*regsize - self.offsety).astype(np.float32)
            bfg = self.trueminf * np.exp(np.random.exponential(scale=1./(self.truealpha-1.),size=nbd)).astype(np.float32)
            # put in function?
            bxxg, bxyg, byyg = self.moments_from_prior(self.truermin_g, nbd)

            # some sources might be generated outside image
            inbounds = (bxg > 0) * (bxg < (imsz[0] - 1)) * (byg > 0) * (byg < (imsz[1] - 1))
            idx_in = np.flatnonzero(inbounds)
            nw = idx_in.size
            bxg = bxg.take(idx_in)
            byg = byg.take(idx_in)
            bfg = bfg.take(idx_in)
            bxxg=bxxg.take(idx_in)
            bxyg=bxyg.take(idx_in)
            byyg=byyg.take(idx_in)
            do_birth_g = True
            factor = np.full(nw, -self.penalty_g)
            goodmove = True
            return do_birth_g, bxg, byg, bfg, bxxg, bxyg, byyg, None, None, None, None, None, None, None, goodmove, factor
        # death
        # does region based death obey detailed balance?
        elif not lifeordeath and self.ng > 0: # need something to kill
            idx_reg = idx_parity(self.xg, self.yg, self.ng, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize)

            nbd = min(nbd, idx_reg.size) # kill nbd sources, or however many sources remain
            nw = nbd
            if nbd > 0:
                idx_kill_g = np.random.choice(idx_reg, size=nbd, replace=False)
                xkg = self.xg.take(idx_kill_g)
                ykg = self.yg.take(idx_kill_g)
                fkg = self.fg.take(idx_kill_g)
                xxkg= self.xxg.take(idx_kill_g)
                xykg= self.xyg.take(idx_kill_g)
                yykg= self.yyg.take(idx_kill_g)
                factor = np.full(nbd, self.penalty_g)
                goodmove = True
                return False, None, None, None, None, None, None, idx_kill_g, xkg, ykg, fkg, xxkg, xykg, yykg, goodmove, factor
            else:
                goodmove = False
                return False, None, None, None, None, None, None, None, None, None, None, None, None, None, goodmove, 0

    def star_galaxy(self):
        starorgalx = np.random.randint(2)
        nsg = (self.nregx * self.nregy) / 4
        # star -> galaxy
        if starorgalx and self.n > 0 and self.ng < self.ngalx:
            idx_reg = idx_parity(self.x, self.y, self.n, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize)
            nsg = min(nsg, min(idx_reg.size, self.ngalx-self.ng))
            nw = nsg
            if nsg > 0:
                idx_kill = np.random.choice(idx_reg, size=nsg, replace=False)
                xk = self.x.take(idx_kill)
                yk = self.y.take(idx_kill)
                fk = self.f.take(idx_kill)
                do_birth_g = True
                bxg = xk.copy()
                byg = yk.copy()
                bfg = fk.copy()

                bxxg, bxyg, byyg = self.moments_from_prior(self.truermin_g, nsg)
                factor = np.full(nsg, self.penalty-self.penalty_g) # TODO factor if star and galaxy flux distributions different
                goodmove = True
                return False, None, None, None, do_birth_g, bxg, byg, bfg, bxxg, bxyg, byyg, idx_kill, xk, yk, fk, None, None, None, None, None, None, None, goodmove, factor
            else:
                goodmove = False
                return False, None, None, None, False, None, None, None, None, None, None, False, None, None, None, None, None, None, None, None, None, None, goodmove, 0
        # galaxy -> star
        elif not starorgalx and self.ng > 1 and self.n < self.nstar:
            idx_reg = idx_parity(self.xg, self.yg, self.ng, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize)
            nsg = min(nsg, min(idx_reg.size, self.nstar-self.n))
            nw = nsg
            if nsg > 0:
                idx_kill_g = np.random.choice(idx_reg, size=nsg, replace=False)
                xkg = self.xg.take(idx_kill_g)
                ykg = self.yg.take(idx_kill_g)
                fkg = self.fg.take(idx_kill_g)
                xxkg= self.xxg.take(idx_kill_g)
                xykg= self.xyg.take(idx_kill_g)
                yykg= self.yyg.take(idx_kill_g)
                do_birth = True
                bx = xkg.copy()
                by = ykg.copy()
                bf = fkg.copy()
                factor = np.full(nsg, self.penalty_g-self.penalty) # TODO factor if star and galaxy flux distributions different
                goodmove = True
                return do_birth, bx, by, bf, False, None, None, None, None, None, None, None, None, None, None, idx_kill_g, xkg, ykg, fkg, xxkg, xykg, yykg, goodmove, factor
            else:
                goodmove = False
                return False, None, None, None, False, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, goodmove, 0


    def twostars_galaxy(self):
        splitsville = np.random.randint(2)
        idx_reg = idx_parity(self.x, self.y, self.n, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize) # stars
        idx_reg_g = idx_parity(self.xg, self.yg, self.ng, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize) # galaxies
        sum_f = 0
        low_n = 0
        idx_bright = idx_reg_g.take(np.flatnonzero(self.fg.take(idx_reg_g) > 2*self.trueminf)) # in region and bright enough to make two stars
        bright_n = idx_bright.size # can only split bright galaxies

        nms = (self.nregx * self.nregy) / 4
        goodmove = False
        # split
        if splitsville and self.ng > 0 and self.n < self.nstar-2 and bright_n > 0: # need something to split, but don't exceed nstar
            nms = min(nms, bright_n, (self.nstar-self.n)/2) # need bright galaxy AND room for split stars
            idx_kill_g = np.random.choice(idx_bright, size=nms, replace=False)
            xkg = self.xg.take(idx_kill_g)
            ykg = self.yg.take(idx_kill_g)
            fkg = self.fg.take(idx_kill_g)
            xxkg= self.xxg.take(idx_kill_g)
            xykg= self.xyg.take(idx_kill_g)
            yykg= self.yyg.take(idx_kill_g)
            fminratio = fkg / self.trueminf # again, care about fmin for stars
            frac = (1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32)
            f1Mf = frac * (1. - frac) # frac(1 - frac)
            agalx, theta, phi = from_moments(xxkg, xykg, yykg)
            dx = agalx * np.cos(phi) / np.sqrt(2 * f1Mf)
            dy = agalx * np.sin(phi) / np.sqrt(2 * f1Mf)
            dr2 = dx*dx + dy*dy
            do_birth = True
            bx = np.column_stack([xkg + ((1-frac)*dx), xkg - frac*dx])
            by = np.column_stack([ykg + ((1-frac)*dy), ykg - frac*dy])
            bf = np.column_stack([fkg * frac, fkg * (1-frac)])

            # don't want to think about how to bounce split-merge
            # don't need to check if above fmin, because of how frac is decided
            inbounds = (bx[:,0] > 0) * (bx[:,0] < imsz[0] - 1) * (by[:,0] > 0) * (by[:,0] < imsz[1] - 1) * \
                       (bx[:,1] > 0) * (bx[:,1] < imsz[0] - 1) * (by[:,1] > 0) * (by[:,1] < imsz[1] - 1)
            idx_in = np.flatnonzero(inbounds)
            xkg = xkg.take(idx_in)
            ykg = ykg.take(idx_in)
            fkg = fkg.take(idx_in)
            xxkg=xxkg.take(idx_in)
            xykg=xykg.take(idx_in)
            yykg=yykg.take(idx_in)
            bx = bx.take(idx_in, axis=0) # birth arrays are 2D
            by = by.take(idx_in, axis=0)
            bf = bf.take(idx_in, axis=0)
            idx_kill_g = idx_kill_g.take(idx_in)
            fminratio = fminratio.take(idx_in)
            frac = frac.take(idx_in)
            dr2 = dr2.take(idx_in)
            f1Mf = f1Mf.take(idx_in)
            theta = theta.take(idx_in)
            goodmove = idx_in.size > 0

            # need star pairs to calculate factor
            sum_f = fkg
            nms = idx_in.size
            nw = nms
            weightoverpairs = np.zeros(nms) # w (1/sum w_1 + 1/sum w_2) / 2
            for k in xrange(nms):
                xtemp = self.x[0:self.n].copy()
                ytemp = self.y[0:self.n].copy()
                xtemp = np.concatenate([xtemp, bx[k:k+1,0], bx[k:k+1,1]])
                ytemp = np.concatenate([ytemp, by[k:k+1,0], by[k:k+1,1]])

                neighi = neighbours(xtemp, ytemp, self.kickrange_g, self.n)
                neighj = neighbours(xtemp, ytemp, self.kickrange_g, self.n+1)
                if neighi > 0 and neighj > 0:
                    weightoverpairs[k] = 1./neighi + 1./neighj
                # else keep zero
            weightoverpairs *= 0.5 * np.exp(-dr2/(2.*self.kickrange_g*self.kickrange_g))
            weightoverpairs[weightoverpairs == 0] = 1
        # merge
        elif not splitsville and idx_reg.size > 1: # need two things to merge!
            nms = min(nms, idx_reg.size/2, self.ngalx-self.ng)
            idx_kill = np.zeros((nms, 2), dtype=np.int)
            choosable = np.zeros(self.nstar, dtype=np.bool)
            choosable[idx_reg] = True
            nchoosable = float(np.count_nonzero(choosable))
            invpairs = np.zeros(nms)
            weightoverpairs = np.zeros(nms)

            for k in xrange(nms):
                idx_kill[k,0] = np.random.choice(self.nstar, p=choosable/nchoosable)
                invpairs[k], idx_kill[k,1] = neighbours(self.x[0:self.n], self.y[0:self.n], self.kickrange_g, idx_kill[k,0], generate=True)
                # prevent sources from being involved in multiple proposals
                if not choosable[idx_kill[k,1]]:
                    idx_kill[k,1] = -1
                if idx_kill[k,1] != -1:
                    invpairs[k] = 1./invpairs[k]
                    invpairs[k] += 1./neighbours(self.x[0:self.n], self.y[0:self.n], self.kickrange_g, idx_kill[k,1])
                    choosable[idx_kill[k,:]] = False
                    nchoosable -= 2
            invpairs *= 0.5

            inbounds = (idx_kill[:,1] != -1)
            idx_in = np.flatnonzero(inbounds)
            nms = idx_in.size
            nw = nms
            idx_kill = idx_kill.take(idx_in, axis=0)
            invpairs = invpairs.take(idx_in)
            goodmove = idx_in.size > 0

            xk = self.x.take(idx_kill) # because idx_kill is 2D so are these
            yk = self.y.take(idx_kill)
            fk = self.f.take(idx_kill)
            dx = xk[:,1] - xk[:,0]
            dy = yk[:,1] - yk[:,0]
            dr2 = dx*dx + dy*dy
            weightoverpairs = np.exp(-dr2/(2.*self.kickrange_g*self.kickrange_g)) * invpairs
            weightoverpairs[weightoverpairs == 0] = 1
            sum_f = np.sum(fk, axis=1)
            fminratio = sum_f / self.trueminf
            frac = fk[:,0] / sum_f
            f1Mf = frac * (1. - frac)
            do_birth_g = True
            bxg = frac*xk[:,0] + (1-frac)*xk[:,1]
            byg = frac*yk[:,0] + (1-frac)*yk[:,1]
            bfg = sum_f
            u = np.random.uniform(low=3e-4, high=1., size=idx_in.size).astype(np.float32) #3e-4 for numerics
            theta = np.arccos(u).astype(np.float32)
            bxxg = f1Mf*(dx*dx+u*u*dy*dy)
            bxyg = f1Mf*(1-u*u)*dx*dy
            byyg = f1Mf*(dy*dy+u*u*dx*dx)
            # this move proposes a splittable galaxy
            bright_n += 1 
        if goodmove:
            factor = 2*np.log(self.truealpha-1) - np.log(self.truealpha_g-1) + 2*(self.truealpha-1)*np.log(self.trueminf) - (self.truealpha_g-1)*np.log(self.trueminf_g) - \
                self.truealpha*np.log(f1Mf) - (2*self.truealpha - self.truealpha_g)*np.log(sum_f) - np.log(imsz[0]*imsz[1]) + np.log(1. - 2./fminratio) - \
                np.log(2*np.pi*self.kickrange_g*self.kickrange_g) + np.log(bright_n/(self.ng+1.-splitsville)) + np.log((self.n-1+2*splitsville)*weightoverpairs) + \
                np.log(sum_f) - np.log(4.) - 2*np.log(dr2) - 3*np.log(f1Mf) - np.log(np.cos(theta)) - 3*np.log(np.sin(theta))
            if not splitsville:
                factor *= -1
                factor += 2*self.penalty - self.penalty_g
                factor += self.log_prior_moments(bxxg, bxyg, byyg)
                return False, None, None, None, do_birth_g, bxg, byg, bfg, bxxg, bxyg, byyg, idx_kill, xk, yk, fk, None, None, None, None, None, None, None, goodmove, factor
            else:
                factor -= 2*self.penalty - self.penalty_g
                factor -= self.log_prior_moments(xxkg, xykg, yykg)
                return do_birth, bx, by, bf, False, None, None, None, None, None, None, None, None, None, None, idx_kill_g, xkg, ykg, fkg, xxkg, xykg, yykg, goodmove, factor
        else:
            return False, None, None, None, False, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, goodmove, 0


    def stargalaxy_galaxy(self):
        splitsville = np.random.randint(2)
        idx_reg_g = idx_parity(self.xg, self.yg, self.ng, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize)
        sum_f = 0
        low_n = 0
        idx_bright = idx_reg_g.take(np.flatnonzero(self.fg.take(idx_reg_g) > self.trueminf + self.trueminf_g)) # in region and bright enough to make s+g
        bright_n = idx_bright.size

        nms = (self.nregx * self.nregy) / 4
        goodmove = False
        # split off star
        if splitsville and self.ng > 0 and self.n < self.nstar and bright_n > 0: # need something to split, but don't exceed nstar
            nms = min(nms, bright_n, self.nstar-self.n) # need bright source AND room for split off star
            dx = (np.random.normal(size=nms)*self.kickrange_g).astype(np.float32)
            dy = (np.random.normal(size=nms)*self.kickrange_g).astype(np.float32)
            idx_move_g = np.random.choice(idx_bright, size=nms, replace=False)
            x0g = self.xg.take(idx_move_g)
            y0g = self.yg.take(idx_move_g)
            f0g = self.fg.take(idx_move_g)
            xx0g = self.xxg.take(idx_move_g)
            xy0g = self.xyg.take(idx_move_g)
            yy0g = self.yyg.take(idx_move_g)
            frac = (self.trueminf_g/f0g + np.random.uniform(size=nms)*(1. - (self.trueminf_g + self.trueminf)/f0g)).astype(np.float32)
            pxg = x0g + ((1-frac)*dx)
            pyg = y0g + ((1-frac)*dy)
            pfg = f0g * frac
            pxxg = (xx0g - frac*(1-frac)*dx*dx)/frac
            pxyg = (xy0g - frac*(1-frac)*dx*dy)/frac
            pyyg = (yy0g - frac*(1-frac)*dy*dy)/frac
            do_birth = True
            bx = x0g - frac*dx
            by = y0g - frac*dy
            bf = f0g * (1-frac)

            # don't want to think about how to bounce split-merge
            # don't need to check if above fmin, because of how frac is decided
            inbounds = (pxg > 0) * (pxg < imsz[0] - 1) * (pyg > 0) * (pyg < imsz[1] - 1) * \
                       (bx > 0) * (bx < imsz[0] - 1) * (by > 0) * (by < imsz[1] - 1) * \
                       (pxxg > 0) * (pyyg > 0) * (pxxg*pyyg > pxyg*pxyg) # TODO minimum galaxy radius?
            idx_in = np.flatnonzero(inbounds)
            x0g = x0g.take(idx_in)
            y0g = y0g.take(idx_in)
            f0g = f0g.take(idx_in)
            xx0g = xx0g.take(idx_in)
            xy0g = xy0g.take(idx_in)
            yy0g = yy0g.take(idx_in)
            pxg = pxg.take(idx_in)
            pyg = pyg.take(idx_in)
            pfg = pfg.take(idx_in)
            pxxg = pxxg.take(idx_in)
            pxyg = pxyg.take(idx_in)
            pyyg = pyyg.take(idx_in)
            bx = bx.take(idx_in)
            by = by.take(idx_in)
            bf = bf.take(idx_in)
            idx_move_g = idx_move_g.take(idx_in)
            frac = frac.take(idx_in)
            goodmove = idx_in.size > 0

            # need to calculate factor
            sum_f = f0g
            nms = idx_in.size
            nw = nms
            invpairs = np.zeros(nms)
            for k in xrange(nms):
                xtemp = self.x[0:self.n].copy()
                ytemp = self.y[0:self.n].copy()
                xtemp = np.concatenate([xtemp, pxg[k:k+1], bx[k:k+1]])
                ytemp = np.concatenate([ytemp, pyg[k:k+1], by[k:k+1]])

                invpairs[k] =  1./neighbours(xtemp, ytemp, self.kickrange_g, self.n)
        # merge star into galaxy
        elif not splitsville and idx_reg_g.size > 1: # need two things to merge!
            nms = min(nms, idx_reg_g.size)
            idx_move_g = np.random.choice(idx_reg_g, size=nms, replace=False) # choose galaxies and then see if they have neighbours
            idx_kill = np.zeros(nms, dtype=np.int)
            choosable = np.full(self.nstar, True, dtype=np.bool)
            nchoosable = float(np.count_nonzero(choosable))
            invpairs = np.zeros(nms)

            for k in xrange(nms):
                l = idx_move_g[k]
                invpairs[k], idx_kill[k] = neighbours(np.concatenate([self.x[0:self.n], self.xg[l:l+1]]), np.concatenate([self.y[0:self.n], self.yg[l:l+1]]), self.kickrange_g, self.n, generate=True)
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
            idx_in = np.flatnonzero(inbounds)
            nms = idx_in.size
            nw = nms
            idx_move_g = idx_move_g.take(idx_in)
            idx_kill = idx_kill.take(idx_in)
            invpairs = invpairs.take(idx_in)
            goodmove = idx_in.size > 0

            x0g = self.xg.take(idx_move_g)
            y0g = self.yg.take(idx_move_g)
            f0g = self.fg.take(idx_move_g)
            xx0g= self.xxg.take(idx_move_g)
            xy0g= self.xyg.take(idx_move_g)
            yy0g= self.yyg.take(idx_move_g)
            xk  = self.x.take(idx_kill)
            yk  = self.y.take(idx_kill)
            fk  = self.f.take(idx_kill)
            sum_f = f0g + fk
            frac = f0g / sum_f
            dx = x0g - xk
            dy = y0g - yk
            pxxg = frac*xx0g + frac*(1-frac)*dx*dx
            pxyg = frac*xy0g + frac*(1-frac)*dx*dy
            pyyg = frac*yy0g + frac*(1-frac)*dy*dy
            inbounds = (pxxg > 0) * (pyyg > 0) * (pxxg*pyyg > pxyg*pxyg) # make sure ellipse is legal TODO min galaxy radius
            x0g = x0g.compress(inbounds)
            y0g = y0g.compress(inbounds)
            f0g = f0g.compress(inbounds)
            xx0g=xx0g.compress(inbounds)
            xy0g=xy0g.compress(inbounds)
            yy0g=yy0g.compress(inbounds)
            xk =   xk.compress(inbounds)
            yk =   yk.compress(inbounds)
            fk =   fk.compress(inbounds)
            sum_f=sum_f.compress(inbounds)
            frac=frac.compress(inbounds)
            pxxg=pxxg.compress(inbounds)
            pxyg=pxyg.compress(inbounds)
            pyyg=pyyg.compress(inbounds)
            goodmove = np.logical_and(goodmove, inbounds.any())

            pxg = frac*x0g + (1-frac)*xk
            pyg = frac*y0g + (1-frac)*yk
            pfg = f0g + fk
            # this proposal makes a galaxy that is bright enough to split
            bright_n = bright_n + 1
        if goodmove:
            factor = np.log(self.truealpha-1) - (self.truealpha-1)*np.log(sum_f/self.trueminf) - self.truealpha_g*np.log(frac) - self.truealpha*np.log(1-frac) + np.log(2*np.pi*self.kickrange_g*self.kickrange_g) - np.log(imsz[0]*imsz[1]) + np.log(1. - (self.trueminf+self.trueminf_g)/sum_f) + np.log(bright_n/float(self.ng)) + np.log((self.n+1.-splitsville)*invpairs) - 2*np.log(frac)
            if not splitsville:
                factor *= -1
            factor += self.log_prior_moments(pxxg, pxyg, pyyg) - self.log_prior_moments(xx0g, xy0g, yy0g) # galaxy prior
            if not splitsville:
                factor += self.penalty
                return idx_move_g, x0g, y0g, f0g, xx0g, xy0g, yy0g, pxg, pyg, pfg, pxxg, pxyg, pyyg, False, None, None, None, idx_kill, xk, yk, fk, goodmove, factor
            else:
                factor -= self.penalty
                return idx_move_g, x0g, y0g, f0g, xx0g, xy0g, yy0g, pxg, pyg, pfg, pxxg, pxyg, pyyg, do_birth, bx, by, bf, None, None, None, None, goodmove, factor
        else:
            return None, None, None, None, None, None, None, None, None, None, None, None, None, False, None, None, None, None, None, None, None, goodmove, 0

    def merge_split_galaxies(self):
        splitsville = np.random.randint(2)
        idx_reg = idx_parity(self.xg, self.yg, self.ng, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize)
        sum_f = 0
        low_n = 0
        idx_bright = idx_reg.take(np.flatnonzero(self.fg.take(idx_reg) > 2*self.trueminf_g)) # in region!
        bright_n = idx_bright.size

        nms = (self.nregx * self.nregy) / 4
        goodmove = False
        # split
        if splitsville and self.ng > 0 and self.ng < self.ngalx and bright_n > 0: # need something to split, but don't exceed nstar
            nms = min(nms, bright_n, self.ngalx-self.ng) # need bright source AND room for split source
            dx = (np.random.normal(size=nms)*self.kickrange_g).astype(np.float32)
            dy = (np.random.normal(size=nms)*self.kickrange_g).astype(np.float32)
            idx_move_g = np.random.choice(idx_bright, size=nms, replace=False)
            x0g = self.xg.take(idx_move_g)
            y0g = self.yg.take(idx_move_g)
            f0g = self.fg.take(idx_move_g)
            xx0g= self.xxg.take(idx_move_g)
            xy0g= self.xyg.take(idx_move_g)
            yy0g= self.yyg.take(idx_move_g)
            fminratio = f0g / self.trueminf_g
            frac = (1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32)
            frac_xx = np.random.uniform(size=nms).astype(np.float32)
            frac_xy = np.random.uniform(size=nms).astype(np.float32)
            frac_yy = np.random.uniform(size=nms).astype(np.float32)
            xx_p = xx0g - frac*(1-frac)*dx*dx# moments of just galaxy pair
            xy_p = xy0g - frac*(1-frac)*dx*dy
            yy_p = yy0g - frac*(1-frac)*dy*dy
            pxg = x0g + ((1-frac)*dx)
            pyg = y0g + ((1-frac)*dy)
            pfg = f0g * frac
            pxxg = xx_p * frac_xx / frac
            pxyg = xy_p * frac_xy / frac
            pyyg = yy_p * frac_yy / frac
            do_birth_g = True
            bxg = x0g - frac*dx
            byg = y0g - frac*dy
            bfg = f0g * (1-frac)
            bxxg = xx_p * (1-frac_xx) / (1-frac) # FIXME is this right?
            bxyg = xy_p * (1-frac_xy) / (1-frac)
            byyg = yy_p * (1-frac_yy) / (1-frac)
            # don't want to think about how to bounce split-merge
            # don't need to check if above fmin, because of how frac is decided
            inbounds = (pxg > 0) * (pxg < imsz[0] - 1) * (pyg > 0) * (pyg < imsz[1] - 1) * \
                       (bxg > 0) * (bxg < imsz[0] - 1) * (byg > 0) * (byg < imsz[1] - 1) * \
                       (pxxg > 0) * (pyyg > 0) * (pxxg*pyyg > pxyg*pxyg) * \
                       (bxxg > 0) * (byyg > 0) * (bxxg*byyg > bxyg*bxyg) # TODO min galaxy rad
            idx_in = np.flatnonzero(inbounds)
            x0g = x0g.take(idx_in)
            y0g = y0g.take(idx_in)
            f0g = f0g.take(idx_in)
            xx0g=xx0g.take(idx_in)
            xy0g=xy0g.take(idx_in)
            yy0g=yy0g.take(idx_in)
            pxg = pxg.take(idx_in)
            pyg = pyg.take(idx_in)
            pfg = pfg.take(idx_in)
            pxxg=pxxg.take(idx_in)
            pxyg=pxyg.take(idx_in)
            pyyg=pyyg.take(idx_in)
            bxg = bxg.take(idx_in)
            byg = byg.take(idx_in)
            bfg = bfg.take(idx_in)
            bxxg=bxxg.take(idx_in)
            bxyg=bxyg.take(idx_in)
            byyg=byyg.take(idx_in)
            idx_move_g = idx_move_g.take(idx_in)
            fminratio = fminratio.take(idx_in)
            frac = frac.take(idx_in)
            xx_p = xx_p.take(idx_in)
            xy_p = xy_p.take(idx_in)
            yy_p = yy_p.take(idx_in)
            goodmove = idx_in.size > 0

            # need to calculate factor
            sum_f = f0g
            nms = idx_in.size
            nw = nms
            invpairs = np.zeros(nms)
            for k in xrange(nms):
                xtemp = self.xg[0:self.ng].copy()
                ytemp = self.yg[0:self.ng].copy()
                xtemp[idx_move_g[k]] = pxg[k]
                ytemp[idx_move_g[k]] = pyg[k]
                xtemp = np.concatenate([xtemp, bxg[k:k+1]])
                ytemp = np.concatenate([ytemp, byg[k:k+1]])

                invpairs[k] =  1./neighbours(xtemp, ytemp, self.kickrange_g, idx_move_g[k])
                invpairs[k] += 1./neighbours(xtemp, ytemp, self.kickrange_g, self.ng)
            invpairs *= 0.5
        # merge
        elif not splitsville and idx_reg.size > 1: # need two things to merge!
            nms = min(nms, idx_reg.size/2)
            idx_move_g = np.zeros(nms, dtype=np.int)
            idx_kill_g = np.zeros(nms, dtype=np.int)
            choosable = np.zeros(self.ngalx, dtype=np.bool)
            choosable[idx_reg] = True
            nchoosable = float(np.count_nonzero(choosable))
            invpairs = np.zeros(nms)

            for k in xrange(nms):
                idx_move_g[k] = np.random.choice(self.ngalx, p=choosable/nchoosable)
                invpairs[k], idx_kill_g[k] = neighbours(self.xg[0:self.ng], self.yg[0:self.ng], self.kickrange_g, idx_move_g[k], generate=True)
                if invpairs[k] > 0:
                    invpairs[k] = 1./invpairs[k]
                # prevent sources from being involved in multiple proposals
                if not choosable[idx_kill_g[k]]:
                    idx_kill_g[k] = -1
                if idx_kill_g[k] != -1:
                    invpairs[k] += 1./neighbours(self.xg[0:self.ng], self.yg[0:self.ng], self.kickrange_g, idx_kill_g[k])
                    choosable[idx_move_g[k]] = False
                    choosable[idx_kill_g[k]] = False
                    nchoosable -= 2
            invpairs *= 0.5

            inbounds = (idx_kill_g != -1)
            idx_in = np.flatnonzero(inbounds)
            nms = idx_in.size
            nw = nms
            idx_move_g = idx_move_g.take(idx_in)
            idx_kill_g = idx_kill_g.take(idx_in)
            invpairs = invpairs.take(idx_in)
            goodmove = idx_in.size > 0

            x0g = self.xg.take(idx_move_g)
            y0g = self.yg.take(idx_move_g)
            f0g = self.fg.take(idx_move_g)
            xx0g= self.xxg.take(idx_move_g)
            xy0g= self.xyg.take(idx_move_g)
            yy0g= self.yyg.take(idx_move_g)
            xkg = self.xg.take(idx_kill_g)
            ykg = self.yg.take(idx_kill_g)
            fkg = self.fg.take(idx_kill_g)
            xxkg= self.xxg.take(idx_kill_g)
            xykg= self.xyg.take(idx_kill_g)
            yykg= self.yyg.take(idx_kill_g)
            sum_f = f0g + fkg
            fminratio = sum_f / self.trueminf_g
            frac = f0g / sum_f

            pxg = frac*x0g + (1-frac)*xkg
            pyg = frac*y0g + (1-frac)*ykg
            pfg = f0g + fkg
            dx = x0g - xkg
            dy = y0g - ykg
            xx_p = frac*xx0g + (1-frac)*xxkg
            xy_p = frac*xy0g + (1-frac)*xykg
            yy_p = frac*yy0g + (1-frac)*yykg
            pxxg = xx_p + frac*(1-frac)*dx*dx
            pxyg = xy_p + frac*(1-frac)*dx*dy
            pyyg = yy_p + frac*(1-frac)*dy*dy

            idx_in = np.flatnonzero((pxxg > 0) * (pyyg > 0) * (pxxg*pyyg > pxyg*pxyg)) # ellipse legal TODO minimum radius
            x0g = x0g.take(idx_in)
            y0g = y0g.take(idx_in)
            f0g = f0g.take(idx_in)
            xx0g=xx0g.take(idx_in)
            xy0g=xy0g.take(idx_in)
            yy0g=yy0g.take(idx_in)
            xkg = xkg.take(idx_in)
            ykg = ykg.take(idx_in)
            fkg = fkg.take(idx_in)
            xxkg=xxkg.take(idx_in)
            xykg=xykg.take(idx_in)
            yykg=yykg.take(idx_in)
            pxg = pxg.take(idx_in)
            pyg = pyg.take(idx_in)
            pfg = pfg.take(idx_in)
            pxxg=pxxg.take(idx_in)
            pxyg=pxyg.take(idx_in)
            pyyg=pyyg.take(idx_in)
            
            idx_move_g = idx_move_g.take(idx_in)
            idx_kill_g = idx_kill_g.take(idx_in)
            invpairs = invpairs.take(idx_in)
            sum_f = sum_f.take(idx_in)
            fminratio = fminratio.take(idx_in)
            frac = frac.take(idx_in)
            xx_p = xx_p.take(idx_in)
            xy_p = xy_p.take(idx_in)
            yy_p = yy_p.take(idx_in)
            
            nms = idx_in.size
            nw = nms
            goodmove = nms > 0
            # turn bright_n into an array
            bright_n = bright_n - (f0g > 2*self.trueminf_g) - (fkg > 2*self.trueminf_g) + (pfg > 2*self.trueminf_g)
        if goodmove:
            factor = np.log(self.truealpha_g-1) + (self.truealpha_g-1)*np.log(self.trueminf) - self.truealpha_g*np.log(frac*(1-frac)*sum_f) + \
                np.log(2*np.pi*self.kickrange_g*self.kickrange_g) - np.log(imsz[0]*imsz[1]) + np.log(1. - 2./fminratio) + np.log(bright_n) + np.log(invpairs) + \
                np.log(sum_f) + np.log(xx_p) + np.log(np.abs(xy_p)) + np.log(yy_p) - 3*np.log(frac) - 3*np.log(1-frac) # last line is Jacobian
            if not splitsville:
                factor *= -1
                factor += self.penalty_g
                factor += self.log_prior_moments(pxxg, pxyg, pyyg) - self.log_prior_moments(xx0g, xy0g, yy0g) - self.log_prior_moments(xxkg, xykg, yykg)
                return idx_move_g, x0g, y0g, f0g, xx0g, xy0g, yy0g, pxg, pyg, pfg, pxxg, pxyg, pyyg, False, None, None, None, None, None, None, idx_kill_g, xkg, ykg, fkg, xxkg, xykg, yykg, goodmove, factor
            else:
                factor -= self.penalty_g
                factor += self.log_prior_moments(pxxg, pxyg, pyyg) + self.log_prior_moments(bxxg, bxyg, byyg) - self.log_prior_moments(xx0g, xy0g, yy0g)
                return idx_move_g, x0g, y0g, f0g, xx0g, xy0g, yy0g, pxg, pyg, pfg, pxxg, pxyg, pyyg, do_birth_g, bxg, byg, bfg, bxxg, bxyg, byyg, None, None, None, None, None, None, None, goodmove, factor
        else:
            return None, None, None, None, None, None, None, None, None, None, None, None, None, False, None, None, None, None, None, None, None, None, None, None, None, None, None, goodmove, 0

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
    xsample[j,:] = models[0].x
    ysample[j,:] = models[0].y
    fsample[j,:] = models[0].f
    ngsample[j] = models[0].ng
    xgsample[j,:] = models[0].xg
    ygsample[j,:] = models[0].yg
    fgsample[j,:] = models[0].fg
    xxgsample[j,:] = models[0].xxg
    xygsample[j,:] = models[0].xyg
    yygsample[j,:] = models[0].yyg

print 'saving...'
np.savez('chain.npz', n=nsample, x=xsample, y=ysample, f=fsample, ng=ngsample, xg=xgsample, yg=ygsample, fg=fgsample, xxg=xxgsample, xyg=xygsample, yyg=yygsample)
