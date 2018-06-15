import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double
import h5py, datetime
# in order for visual=True to work, interactive backend should be loaded before importing pyplot
import matplotlib 
import seaborn as sns
sns.set(context='poster', style='ticks', color_codes=True)

import cPickle

import scipy.spatial
import networkx as nx

import time
import astropy.wcs
import astropy.io.fits

import sys, os, warnings

from galaxy import to_moments, from_moments, retr_sers, retr_tranphon

from __init__ import *

def eval_modl(gdat, x, y, f, back, numbpixlpsfnside, coefspix, lcpreval, \
                                            regsize=None, margin=0, offsetx=0, offsety=0, weig=None, ref=None, lib=None, sizeimag=None):
    
    assert x.dtype == np.float32
    assert y.dtype == np.float32
    assert f.dtype == np.float32
    assert coefspix.dtype == np.float32
    if ref is not None:
        assert ref.dtype == np.float32
    
    if sizeimag == None:
        sizeimag = gdat.sizeimag

    numbpixlpsfn = numbpixlpsfnside**2

    numbparaspix = coefspix.shape[0]

    if weig is None:
        if gdat.booltimebins:
            weig = np.full([gdat.numbtime] + sizeimag, 1., dtype=np.float32)
        else:
            weig = np.full(sizeimag, 1., dtype=np.float32)
    if regsize is None:
        regsize = max(sizeimag[0], sizeimag[1])

    # FIXME sometimes phonions are outside image... what is best way to handle?
    goodsrc = (x > 0) * (x < sizeimag[0] - 1) * (y > 0) * (y < sizeimag[1] - 1)
    x = x.compress(goodsrc)
    y = y.compress(goodsrc)
    f = f.compress(goodsrc)

    numbphon = x.size
    rad = numbpixlpsfnside / 2 # 12 for numbpixlpsfnside = 25

    numbregiyaxi = sizeimag[1] / regsize + 1 # assumes sizeimag % regsize = 0?
    numbregixaxi = sizeimag[0] / regsize + 1

    ix = np.ceil(x).astype(np.int32)
    dx = ix - x
    iy = np.ceil(y).astype(np.int32)
    dy = iy - y

    dd = np.column_stack((np.full(numbphon, 1., dtype=np.float32), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, dx*dx*dy, dx*dy*dy, dy*dy*dy)).astype(np.float32) * f[:, None]

    if lib is None:
        
        modl = np.full((sizeimag[1]+2*rad+1,sizeimag[0]+2*rad+1), back, dtype=np.float32)
        recon2 = np.dot(dd, coefspix).reshape((numbphon,numbpixlpsfnside,numbpixlpsfnside))
        recon = np.zeros((numbphon,numbpixlpsfnside,numbpixlpsfnside), dtype=np.float32)
        recon[:,:,:] = recon2[:,:,:]
        for i in xrange(numbphon):
            modl[iy[i]:iy[i]+rad+rad+1,ix[i]:ix[i]+rad+rad+1] += recon[i,:,:]

        modl = modl[rad:sizeimag[1]+rad,rad:sizeimag[0]+rad]

        if ref is not None:
                diff = ref - modl
        diff2 = np.zeros((numbregiyaxi, numbregixaxi), dtype=np.float64)
        for i in xrange(numbregiyaxi):
            y0 = max(i*regsize - offsety - margin, 0)
            y1 = min((i+1)*regsize - offsety + margin, sizeimag[1])
            for j in xrange(numbregixaxi):
                x0 = max(j*regsize - offsetx - margin, 0)
                x1 = min((j+1)*regsize - offsetx + margin, sizeimag[0])
                subdiff = diff[y0:y1,x0:x1]
                diff2[i,j] = np.sum(subdiff*subdiff*weig[y0:y1,x0:x1])
    else:
        recon = np.zeros((numbphon, numbpixlpsfn), dtype=np.float32)
        reftemp = ref
        if ref is None:
            if gdat.booltimebins:
                reftemp = np.zeros((gdat.numbtime, sizeimag[1], sizeimag[0]), dtype=np.float32)
            else:
                reftemp = np.zeros((sizeimag[1], sizeimag[0]), dtype=np.float32)
        if gdat.booltimebins:
            modl = np.full((gdat.numbtime, sizeimag[1], sizeimag[0]), back, dtype=np.float32)
        else:
            modl = np.full((sizeimag[1], sizeimag[0]), back, dtype=np.float32)
        diff2 = np.zeros((numbregiyaxi, numbregixaxi), dtype=np.float64)
        
        if gdat.verbtype > 1:
            print 'dd'
            summgene(dd)
            print 'coefspix'
            summgene(coefspix)
            print 'recon'
            summgene(recon)
            print 'iy'
            print iy
            print 'modl'
            summgene(modl)
            print 'reftemp'
            summgene(reftemp)
            print 'weig'
            print weig.shape
            print 'diff2'
            print diff2.shape
            print 'gdat.numbtime'
            print gdat.numbtime
            print 'gdat.booltimebins'
            print gdat.booltimebins
            print 'lcpreval'
            summgene(lcpreval)
            print

        lib(sizeimag[0], sizeimag[1], numbphon, numbpixlpsfnside, numbparaspix, dd, coefspix, recon, ix, iy, modl, \
                                                reftemp, weig, diff2, regsize, margin, offsetx, offsety, gdat.numbtime, gdat.booltimebins, lcpreval)

    if ref is not None:
        return modl, diff2
    else:
        return modl


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


# visualization-related functions
def setp_imaglimt(gdat, axis):
    
    axis.set_xlim(-0.5, gdat.sizeimag[0] - 0.5)
    axis.set_ylim(-0.5, gdat.sizeimag[1] - 0.5)
                    
    
def idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, regsize):
    match_x = (get_region(x[0:n], offsetx, regsize) % 2) == parity_x
    match_y = (get_region(y[0:n], offsety, regsize) % 2) == parity_y
    return np.flatnonzero(np.logical_and(match_x, match_y))


def retr_numbdoff(numbstar, numbgalx, numbener):
    
    numbdoff = (2 + numbener) * numbstar + (5 + numbener) * numbgalx

    return numbdoff


def make_cmapdivg(strgcolrloww, strgcolrhigh):

    funccolr = matplotlib.colors.ColorConverter().to_rgb
   
    colrloww = funccolr(strgcolrloww)
    colrhigh = funccolr(strgcolrhigh)
   
    cmap = make_cmap([colrloww, funccolr('white'), 0.5, funccolr('white'), colrhigh])

    return cmap


def make_cmap(seq):
    
    sequ = [(None,) * 3, 0.] + list(seq) + [1.0, (None,) * 3]
    colrdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(sequ):
        if isinstance(item, float):
            red1, gre1, blu1 = sequ[i - 1]
            red2, gre2, blu2 = sequ[i + 1]
            colrdict['red'].append([item, red1, red2])
            colrdict['green'].append([item, gre1, gre2])
            colrdict['blue'].append([item, blu1, blu2])
   
    return matplotlib.colors.LinearSegmentedColormap('CustomMap', colrdict)


class gdatstrt(object):

    def __init__(self):
        self.boollockmodi = False
        pass
    
    def __setattr__(self, attr, valu):
        super(gdatstrt, self).__setattr__(attr, valu)


#def summgene(varb):
#
#    print np.amin(varb)
#    print np.amax(varb)
#    print np.mean(varb)
#    print varb.shape


def main( \
         # string characterizing the type of data (experiment that collected it and its PSF, etc.)
         strgdata='sdss.0921', \
        
         # a Boolean flag indicating whether the data is time-binned
         booltimebins=False, \

         # number of samples
         numbsamp=100, \
    
         # number of samples
         numbsampburn=None, \
    
         # number of loops
         numbloop=1000, \
         
         # string indicating whether the data is simulated or input by the user
         # 'mock' for mock data, 'inpt' for input data
         datatype='mock', \

         # boolean flag whether to show the image and catalog samples interactively
         boolplotshow=False, \
         
         # boolean flag whether to save the image and catalog samples to disc
         boolplotsave=True, \
        
         # a string extension to the run tag
         rtagextn=None, \
       
         # level of verbosity
         verbtype=1, \

         # color style
         colrstyl='lion', \

         # boolean flag whether to test the PSF
         testpsfn=False, \
         
         # string to indicate the ingredients of the photometric model
         # 'star' for star only, 'stargalx' for star and galaxy
         strgmodl='star', \
         ):

    # construct the global object 
    gdat = gdatstrt()
    for attr, valu in locals().iteritems():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # check if source has been changed after compilation
    if os.path.getmtime('blas.c') > os.path.getmtime('blas.so'):
        warnings.warn('blas.c modified after compiled blas.so', Warning)
    
    #np.seterr(all='raise')

    # load arguments into the global object
    #gdat.boolplotsave = boolplotsave
    #gdat.booltimebins = booltimebins
    #gdat.verbtype = booltimebins

    # time stamp string
    strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
   
    # run tag
    rtag = strgtimestmp 
    if rtagextn != None:
        rtag += '_' + rtagextn

    if numbsampburn == None:
        numbsampburn = int(0.2 * numbsamp)

    print 'Lion initialized at %s' % strgtimestmp

    # show critical inputs
    print 'strgdata: ', strgdata
    print 'Model type:', strgmodl
    print 'Data type:', datatype
    
    cmapresi = make_cmapdivg('Red', 'Orange')
    if colrstyl == 'pcat':
        sizemrkr = 1e-2
        linewdth = 3
        histtype = 'bar'
        colrbrgt = 'green'
    else:
        linewdth = None
        sizemrkr = 1. / 1360.
        histtype = 'step'
        colrbrgt = 'lime'
    
    if boolplotshow:
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if boolplotshow:
        plt.ion()
    
    # ix, iy = 0. to 3.999
    def plot_psfn(gdat, nc, cf, psf, ix, iy, lib=None):
        
        lcpreval = np.array([[0.]], dtype=np.float32)
        xpos = np.array([12. - ix / 5.], dtype=np.float32)
        ypos = np.array([12. - iy / 5.], dtype=np.float32)
        flux = np.array([1.], dtype=np.float32)
        back = 0.
        sizeimag = [25, 25]
        psf0 = eval_modl(gdat, xpos, ypos, flux, back, nc, cf, lcpreval, lib=lib, sizeimag=sizeimag)
        plt.subplot(2,2,1)
        if gdat.booltimebins:
            temp = psf0[0, :, :]
        else:
            temp = psf0
        plt.imshow(temp, interpolation='none', origin='lower')
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
        plt.subplot(2, 2, 3)
        plt.title('absolute difference')
        plt.imshow(temp - realpsf, interpolation='none', origin='lower')
        plt.colorbar()
        plt.subplot(2,2,4)
        plt.imshow((temp - realpsf) * invrealpsf, interpolation='none', origin='lower')
        plt.colorbar()
        plt.title('fractional difference')
        if gdat.boolplotsave:
            plt.savefig(pathdatartag + '%s_psfn.pdf' % strgtimestmp)
        else:
            plt.show()
    
    def psf_poly_fit(gdat, psfnusam, factusam):
        
        assert psfnusam.shape[0] == psfnusam.shape[1] # assert PSF is square
        
        # number of pixels along the side of the upsampled PSF
        numbpixlpsfnsideusam = psfnusam.shape[0]
    
        # pad by one row and one column
        psfnusampadd = np.zeros((numbpixlpsfnsideusam+1, numbpixlpsfnsideusam+1), dtype=np.float32)
        psfnusampadd[0:numbpixlpsfnsideusam, 0:numbpixlpsfnsideusam] = psfnusam
    
        # make design matrix for each factusam x factusam region
        numbpixlpsfnside = numbpixlpsfnsideusam / factusam # dimension of original psf
        nx = factusam + 1
        y, x = np.mgrid[0:nx, 0:nx] / np.float32(factusam)
        x = x.flatten()
        y = y.flatten()
        A = np.column_stack([np.full(nx*nx, 1, dtype=np.float32), x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y]).astype(np.float32)
        
        # number of subpixel parameters
        numbparaspix = A.shape[1]
    
        # output array of coefficients
        coefspix = np.zeros((numbparaspix, numbpixlpsfnside, numbpixlpsfnside), dtype=np.float32)
    
        # loop over original psf pixels and get fit coefficients
        for i in xrange(numbpixlpsfnside):
            for j in xrange(numbpixlpsfnside):
                # solve p = A coefspix for coefspix
                p = psfnusampadd[i*factusam:(i+1)*factusam+1, j*factusam:(j+1)*factusam+1].flatten()
                coefspix[:, i, j] = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, p)) 
        coefspix = coefspix.reshape(coefspix.shape[0], coefspix.shape[1] * coefspix.shape[2])
        
        if gdat.boolplotsave:
            figr, axis = plt.subplots()
            axis.imshow(psfnusampadd, interpolation='none')
            plt.savefig(pathdatartag + '%s_psfnusampadd.pdf' % strgtimestmp)
    
        return coefspix
   
    pathlion = os.environ['LION_PATH'] + '/'
    pathliondata = os.environ["LION_PATH"] + '/Data/'
    pathdata = os.environ['LION_DATA_PATH'] + '/'

    pathdatartag = pathdata + rtag + '/'
    os.system('mkdir -p %s' % pathdatartag)

    # constants
    ## number of bands (i.e., energy bins)
    numbener = 1
    
    boolplot = boolplotshow or boolplotsave

    # read PSF
    f = open(pathliondata + strgdata+'_psf.txt')
    nc, nbin = [np.int32(i) for i in f.readline().split()]
    print 'nc'
    print nc
    print 'nbin'
    print nbin
    f.close()
    psf = np.loadtxt(pathliondata + strgdata+'_psf.txt', skiprows=1).astype(np.float32)
    cf = psf_poly_fit(gdat, psf, nbin)
    npar = cf.shape[0]
    
    # construct C library
    array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
    array_3d_float = npct.ndpointer(dtype=np.float32, ndim=3, flags="C_CONTIGUOUS")
    array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    array_2d_double = npct.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
    array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")
    libmmult = npct.load_library('blas', '.')
    libmmult.pcat_model_eval.restype = None
    
    #void pcat_model_eval(int NX, int NY, int nstar, int nc, int k, float* A, float* B, float* C, int* x,
	#int* y, float* image, float* ref, float* weig, double* diff2, int regsize, int margin, int offsetx, int offsety)
    
    #lib(gdat.sizeimag[0], gdat.sizeimag[1], nstar, nc, cf.shape[0], dd, cf, recon, ix, iy, image, reftemp, weig, diff2, regsize, margin, offsetx, offsety)
    
    #libmmult.pcat_model_eval.argtypes = [c_int NX gdat.sizeimag[0], c_int NY gdat.sizeimag[1], c_int nstar, c_int nc, c_int k cf.shape[0]
    
    # array_2d_float A dd, array_2d_float B cf, array_2d_float C recon
    # array_1d_int x ix, array_1d_int y iy, array_2d_float image, array_2d_float ref reftemp, array_2d_float weig weig, array_2d_double diff2, c_int regsize, 
    # c_int margin, c_int offsetx, c_int offsety]
    
    libmmult.pcat_imag_acpt.restype = None
    libmmult.pcat_like_eval.restype = None
    libmmult.pcat_model_eval.argtypes = [c_int, c_int, c_int, c_int, c_int]
    libmmult.pcat_imag_acpt.argtypes = [c_int, c_int]
    libmmult.pcat_like_eval.argtypes = [c_int, c_int]
    
    libmmult.pcat_model_eval.argtypes += [array_2d_float, array_2d_float, array_2d_float]
    
    libmmult.pcat_model_eval.argtypes += [array_1d_int, array_1d_int]
    
    if gdat.booltimebins:
        libmmult.pcat_imag_acpt.argtypes += [array_3d_float, array_3d_float]
        libmmult.pcat_model_eval.argtypes += [array_3d_float, array_3d_float, array_3d_float]
        libmmult.pcat_like_eval.argtypes += [array_3d_float, array_3d_float, array_3d_float]
    else:
        libmmult.pcat_imag_acpt.argtypes += [array_2d_float, array_2d_float]
        libmmult.pcat_model_eval.argtypes += [array_2d_float, array_2d_float, array_2d_float]
        libmmult.pcat_like_eval.argtypes += [array_2d_float, array_2d_float, array_2d_float]
    
    libmmult.pcat_model_eval.argtypes += [array_2d_double]
    libmmult.pcat_like_eval.argtypes += [array_2d_double]
    
    libmmult.pcat_model_eval.argtypes += [c_int, c_int, c_int, c_int, c_int, c_int, array_2d_float]
    libmmult.pcat_imag_acpt.argtypes += [array_2d_int, c_int, c_int, c_int, c_int]
    libmmult.pcat_like_eval.argtypes += [c_int, c_int, c_int, c_int]
    
    # read data
    f = open(pathliondata + strgdata+'_pix.txt')
    w, h, nband = [np.int32(i) for i in f.readline().split()]
    gdat.sizeimag = (w, h)
    assert nband == 1
    bias, gain = [np.float32(i) for i in f.readline().split()]
    f.close()
    data = np.loadtxt(pathliondata + strgdata+'_cts.txt').astype(np.float32)
    data -= bias
    trueback = np.float32(179)#np.float32(445*250)#179.)
    numbpixl = w * h
    
    if gdat.booltimebins:
        gdat.numbtime = 10
        numblcpr = gdat.numbtime - 1
        data = np.tile(data, (gdat.numbtime, 1, 1))
        indxtime = np.arange(gdat.numbtime)
        print 'data'
        summgene(data)
        for k in indxtime:
            data[k, :, :] += (4 * np.random.randn(w * h).reshape((w, h))).astype(int)
        print 'data'
        summgene(data)
    else:
        gdat.numbtime = 1
    gdat.indxtime = np.arange(gdat.numbtime)

    print 'Image width and height: %d %d pixels' % (w, h)
    
    # plots
    ## PSF
    if boolplot and testpsfn:
        plot_psfn(gdat, nc, cf, psf, np.float32(np.random.uniform()*4), np.float32(np.random.uniform()*4), libmmult.pcat_model_eval)
   
    ## data
    if False and gdat.booltimebins:
        for k in gdat.indxtime:
            figr, axis = plt.subplots()
            axis.imshow(data[k, :, :], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data), vmax=np.percentile(data, 95))
            ## limits
            setp_imaglimt(gdat, axis)
            plt.savefig(pathdatartag + '%s_cntpdatainit%04d.pdf' % (strgtimestmp, k))
        print 'Making the animation...'
        cmnd = 'convert -delay 20 -density 200x200 %s/%s_cntpdatainit*.pdf %s/%s_cntpdatainit.gif' % (pathdatartag, strgtimestmp, pathdatartag, strgtimestmp)
        print cmnd
        os.system(cmnd)
    
    variance = data / gain
    weig = 1. / variance # inverse variance
   
    gdat.indxxpos = 0
    gdat.indxypos = 1
    gdat.indxflux = 2
    gdat.numbparastar = 3
    if gdat.booltimebins:
        gdat.indxlcpr = np.arange(gdat.numbparastar, gdat.numbparastar + numblcpr)
        gdat.numbparastar += numblcpr
    if 'galx' in strgmodl:
        _XX = 3
        _XY = 4
        _YY = 5
            
    class Proposal:
    
        #_X = 0
        #_Y = 1
        #_F = 2
        #numbparastar = 3
        #if gdat.booltimebins:
        #    _L = np.arange(numbparastar, numbparastar + numblcpr)
        #    numbparastar += numblcpr
        #if 'galx' in strgmodl:
        #    _XX = 3
        #    _XY = 4
        #    _YY = 5
            
        gridphon, amplphon = retr_sers(sersindx=2.)
        
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
            self.xphon = np.append(self.xphon, stars[gdat.indxxpos,:])
            self.yphon = np.append(self.yphon, stars[gdat.indxypos,:])
            self.fphon = np.append(self.fphon, fluxmult*stars[gdat.indxflux,:])
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
                return self.stars0[gdat.indxxpos,:], self.stars0[gdat.indxypos,:]
            elif self.idx_move_g is not None:
                return self.galaxies0[gdat.indxxpos,:], self.galaxies0[gdat.indxypos,:]
            elif self.do_birth:
                bx, by = self.starsb[[gdat.indxxpos,gdat.indxypos],:]
                refx = bx if bx.ndim == 1 else bx[:,0]
                refy = by if by.ndim == 1 else by[:,0]
                return refx, refy
            elif self.do_birth_g:
                return self.galaxiesb[gdat.indxxpos,:], self.galaxiesb[gdat.indxypos,:]
            elif self.idx_kill is not None:
                xk, yk = self.starsk[[gdat.indxxpos,gdat.indxypos],:]
                refx = xk if xk.ndim == 1 else xk[:,0]
                refy = yk if yk.ndim == 1 else yk[:,0]
                return refx, refy
            elif self.idx_kill_g is not None:
                return self.galaxiesk[gdat.indxxpos,:], self.galaxiesk[gdat.indxypos,:]

    def supr_catl(axis, xpos, ypos, flux, xpostrue=None, ypostrue=None, fluxtrue=None):
        
        if datatype == 'mock':
            if strgmodl == 'star' or strgmodl == 'galx':
                mask = fluxtrue > 250
                axis.scatter(xpostrue[mask], ypostrue[mask], marker='+', s=sizemrkr*fluxtrue[mask], color=colrbrgt, lw=2)
                mask = np.logical_not(mask)
                axis.scatter(xpostrue[mask], ypostrue[mask], marker='+', s=sizemrkr*fluxtrue[mask], color='g', lw=2)
        if colrstyl == 'pcat':
            colr = 'b'
        else:
            colr = 'r'
        axis.scatter(xpos, ypos, marker='x', s=sizemrkr*flux, color=colr, lw=2)
    
    
    class Model:
        
        # should these be class or instance variables?
        nstar = 2000
        trueminf = np.float32(250.)#*136)
        truealpha = np.float32(2.00)
        back = trueback
    
        ngalx = 100
        trueminf_g = np.float32(250.)#*136)
        truealpha_g = np.float32(2.00)
        truermin_g = np.float32(1.00)
    
        gridphon, amplphon = retr_sers(sersindx=2.)
    
        penalty = 1.5
        penalty_g = 3.0
        kickrange = 1.
        kickrange_g = 1.
        
        #_X = 0
        #_Y = 1
        #_F = 2
        #numbparastar = 3
        #if gdat.booltimebins:
        #    _L = np.arange(numbparastar, numbparastar + numblcpr)
        #    numbparastar += numblcpr
        #if 'galx' in strgmodl:
        #    _XX = 3
        #    _XY = 4
        #    _YY = 5
        

        def __init__(self):
            self.n = np.random.randint(self.nstar)+1
            self.stars = np.zeros((gdat.numbparastar, self.nstar), dtype=np.float32)
            self.stars[:, :self.n] = np.random.uniform(size=(gdat.numbparastar, self.n))  # refactor into some sort of prior function?
            self.stars[gdat.indxxpos,0:self.n] *= gdat.sizeimag[0]-1
            self.stars[gdat.indxypos,0:self.n] *= gdat.sizeimag[1]-1
            self.stars[gdat.indxflux,0:self.n] **= -1./(self.truealpha - 1.)
            self.stars[gdat.indxflux,0:self.n] *= self.trueminf
    
            self.ng = 0
            if strgmodl == 'galx':
                self.ng = np.random.randint(self.ngalx)+1
                self.galaxies = np.zeros((6,self.ngalx), dtype=np.float32)
                # temp -- 3 should be generalized to temporal modeling
                self.galaxies[[gdat.indxxpos,gdat.indxypos,gdat.indxflux],0:self.ng] = np.random.uniform(size=(3, self.ng))
                self.galaxies[gdat.indxxpos,0:self.ng] *= gdat.sizeimag[0]-1
                self.galaxies[gdat.indxypos,0:self.ng] *= gdat.sizeimag[1]-1
                self.galaxies[gdat.indxflux,0:self.ng] **= -1./(self.truealpha_g - 1.)
                self.galaxies[gdat.indxflux,0:self.ng] *= self.trueminf_g
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
    
        
        def run_sampler(self, gdat, temperature, jj, boolplotshow=False):
            
            if gdat.verbtype > 1:
                print 'Sample %d started.' % jj
            
            t0 = time.clock()
            nmov = np.zeros(gdat.numbloop)
            movetype = np.zeros(gdat.numbloop)
            accept = np.zeros(gdat.numbloop)
            outbounds = np.zeros(gdat.numbloop)
            dt1 = np.zeros(gdat.numbloop)
            dt2 = np.zeros(gdat.numbloop)
            dt3 = np.zeros(gdat.numbloop)
    
            self.offsetx = np.random.randint(regsize)
            self.offsety = np.random.randint(regsize)
            self.nregx = gdat.sizeimag[0] / regsize + 1
            self.nregy = gdat.sizeimag[1] / regsize + 1
    
            resid = data.copy() # residual for zero image is data
            lcpreval = np.array([[0.]], dtype=np.float32)
            if strgmodl == 'star':
                xposeval = self.stars[gdat.indxxpos,0:self.n]
                yposeval = self.stars[gdat.indxypos,0:self.n]
                fluxeval = self.stars[gdat.indxflux,0:self.n]
                if gdat.booltimebins:
                    lcpreval = self.stars[gdat.indxlcpr, :self.n]
            else:
                xposphon, yposphon, specphon = retr_tranphon(self.gridphon, self.amplphon, self.galaxies[:,0:self.ng])
                xposeval = np.concatenate([self.stars[gdat.indxxpos,0:self.n], xposphon]).astype(np.float32)
                yposeval = np.concatenate([self.stars[gdat.indxypos,0:self.n], yposphon]).astype(np.float32)
                fluxeval = np.concatenate([self.stars[gdat.indxflux,0:self.n], specphon]).astype(np.float32)
            numbphon = xposeval.size
            model, diff2 = eval_modl(gdat, xposeval, yposeval, fluxeval, self.back, nc, cf, lcpreval, \
                                                               weig=weig, ref=resid, lib=libmmult.pcat_model_eval, \
                                                               regsize=regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety)
            logL = -0.5*diff2
            resid -= model
    
            moveweig = np.array([80., 40., 40.])
            if strgmodl == 'galx':
                moveweig = np.array([80., 40., 40., 80., 40., 40., 40., 40., 40.])
            moveweig /= np.sum(moveweig)
    
            for i in xrange(gdat.numbloop):

                if gdat.verbtype > 1:
                    print 'Loop %d started.' % i
                
                t1 = time.clock()
                rtype = np.random.choice(moveweig.size, p=moveweig)
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
                    dmodel, diff2 = eval_modl(gdat, proposal.xphon, proposal.yphon, proposal.fphon, dback, nc, cf, lcpreval, \
                                weig=weig, ref=resid, lib=libmmult.pcat_model_eval, regsize=regsize, margin=margin, offsetx=self.offsetx, offsety=self.offsety)
                    plogL = -0.5*diff2
                    dt2[i] = time.clock() - t2
    
                    t3 = time.clock()
                    refx, refy = proposal.get_ref_xy()
                    regionx = get_region(refx, self.offsetx, regsize)
                    regiony = get_region(refy, self.offsety, regsize)
    
                    ###
                    '''
                    if i == 0:
                        yy, xx = np.mgrid[0:100,0:100]
                        rxx = get_region(xx, self.offsetx, regsize) % 2
                        ryy = get_region(yy, self.offsety, regsize) % 2
                        rrr = rxx*2 + ryy
                        plt.figure(2)
                        plt.imshow(rrr, interpolation='none', origin='lower', cmap='Accent')
                        plt.savefig('region-'+str(int(time.clock()*10))+'.pdf')
                        plt.show()
                        plt.figure(3)
                        vmax=np.max(np.abs(dmodel))
                        plt.imshow(dmodel, interpolation='none', origin='lower', cmap='bwr', vmin=-vmax, vmax=vmax)
                        plt.savefig('dmodel-'+str(int(time.clock()*10))+'.pdf')
                        plt.show()
                    '''
                    ###
    
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
                    libmmult.pcat_imag_acpt(gdat.sizeimag[0], gdat.sizeimag[1], dmodel, dmodel_acpt, acceptreg, regsize, margin, self.offsetx, self.offsety)
                    # using this dmodel containing only accepted moves, update logL
                    diff2.fill(0)
                    libmmult.pcat_like_eval(gdat.sizeimag[0], gdat.sizeimag[1], dmodel_acpt, resid, weig, diff2, regsize, margin, self.offsetx, self.offsety)
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
                        
                        if gdat.verbtype > 1:
                            print 'proposal.starsb'
                            summgene(proposal.starsb)
                        
                        starsb = proposal.starsb.compress(acceptprop, axis=1)
                        
                        if gdat.verbtype > 1:
                            print 'starsb'
                            summgene(starsb)
                        
                        starsb = starsb.reshape((gdat.numbparastar, -1))
                        numbbrth = starsb.shape[1]
                        
                        if gdat.verbtype > 1:
                            print 'starsb'
                            summgene(starsb)
                            print 'self.n'
                            print self.n
                            print 'numbbrth'
                            print numbbrth
                            print 'self.stars'
                            summgene(self.stars)
                            print
                        
                        self.stars[:, self.n:self.n+numbbrth] = starsb
                        self.n += numbbrth
                    if proposal.do_birth_g:
                        galaxiesb = proposal.galaxiesb.compress(acceptprop, axis=1)
                        numbbrth = galaxiesb.shape[1]
                        self.galaxies[:, self.ng:self.ng+numbbrth] = galaxiesb
                        self.ng += numbbrth
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
            
            
                if gdat.verbtype > 1:
                    print 'diff2'
                    print diff2
                    print 'logL'
                    print logL
                    print 'Loop %d ended.' % i
                    print
                    print
                    print

            numbdoff = retr_numbdoff(self.n, self.ng, numbener)
            chi2 = np.sum(weig*(data-model)*(data-model))
            chi2doff = chi2 / (numbpixl - numbdoff)
            fmtstr = '\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f (Pg) %0.3f (BDg) %0.3f (S-g) %0.3f (gSg) %0.3f (gMS) %0.3f'
            print 'Temperature', temperature, 'background', self.back, 'N_star', self.n, 'N_gal', self.ng, 'N_phon', numbphon, \
                                                                                            'chi^2', chi2, 'numbdoff', numbdoff, 'chi2dof', chi2doff
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
            
            if gdat.verbtype > 1:
                print 'Sample %d ended' % jj
                print
                print
                print
                print
                print
                print
                print
                print
                print
    
            if boolplotshow or boolplotsave:
                
                if boolplotshow:
                    plt.figure(1)
                    plt.clf()
                    plt.subplot(1, 3, 1)
                    plt.imshow(data, origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data), vmax=np.percentile(data, 95))
                    
                    # overplot point sources
                    if datatype == 'mock':
                        if strgmodl == 'galx':
                            plt.scatter(truexg, trueyg, marker='1', s=sizemrkr*truefg, color='lime')
                            plt.scatter(truexg, trueyg, marker='o', s=truerng*truerng*4, edgecolors='lime', facecolors='none')
                        if strgmodl == 'stargalx':
                            plt.scatter(gdat.truexposstar, ypostrue[mask], marker='+', s=np.sqrt(fluxtrue[mask]), color='g')
                    if strgmodl == 'galx':
                        plt.scatter(self.galaxies[gdat.indxxpos, 0:self.ng], self.galaxies[gdat.indxypos, 0:self.ng], marker='2', s=sizemrkr*self.galaxies[gdat.indxflux, 0:self.ng], color='r')
                        a, theta, phi = from_moments(self.galaxies[self._XX, 0:self.ng], self.galaxies[self._XY, 0:self.ng], self.galaxies[self._YY, 0:self.ng])
                        plt.scatter(self.galaxies[gdat.indxxpos, 0:self.ng], self.galaxies[gdat.indxypos, 0:self.ng], marker='o', s=4*a*a, edgecolors='red', facecolors='none')
                    
                    supr_catl(plt.gca(), self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.stars[gdat.indxflux, 0:self.n], xpostrue, ypostrue, fluxtrue)
                    
                    plt.subplot(1, 3, 2)
                    
                    ## residual plot
                    if colrstyl == 'pcat':
                        cmap = cmapresi
                    else:
                        cmap = 'bwr'
                    plt.imshow(resid*np.sqrt(weig), origin='lower', interpolation='none', vmin=-5, vmax=5, cmap=cmap)
                    if j == 0:
                        plt.tight_layout()
                    
                    
                    plt.subplot(1, 3, 3)
    
                    ## flux histogram
                    if datatype == 'mock':
                        if colrstyl == 'pcat':
                            colr = 'g'
                        else:
                            colr = None
                        plt.hist(np.log10(fluxtrue), range=(np.log10(self.trueminf), np.log10(np.max(fluxtrue))), \
                                                                                        log=True, alpha=0.5, label=labldata, histtype=histtype, lw=linewdth, \
                                                                                        color=colr, facecolor=colr, edgecolor=colr)
                        if colrstyl == 'pcat':
                            colr = 'b'
                        else:
                            colr = None
                        plt.hist(np.log10(self.stars[gdat.indxflux, 0:self.n]), range=(np.log10(self.trueminf), \
                                                                                        np.log10(np.max(fluxtrue))), color=colr, facecolor=colr, lw=linewdth, \
                                                                                        log=True, alpha=0.5, label='Sample', histtype=histtype, edgecolor=colr)
                    else:
                        if colrstyl == 'pcat':
                            colr = 'b'
                        else:
                            colr = None
                        plt.hist(np.log10(self.stars[gdat.indxflux, 0:self.n]), range=(np.log10(self.trueminf), \
                                                                             np.ceil(np.log10(np.max(self.stars[gdat.indxflux, 0:self.n])))), lw=linewdth, \
                                                                             facecolor=colr, color=colr, log=True, alpha=0.5, label='Sample', histtype=histtype, edgecolor=colr)
                    plt.legend()
                    plt.xlabel('log10 flux')
                    plt.ylim((0.5, self.nstar))
                    
                    plt.draw()
                    plt.pause(1e-5)
                    
                else:
                    
                    # count map
                    figr, axis = plt.subplots()
                    if gdat.booltimebins:
                        temp = data[0, :, :]
                    else:
                        temp = data
                    axis.imshow(temp, origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data), vmax=np.percentile(data, 95))
                    ## overplot point sources
                    supr_catl(axis, self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.stars[gdat.indxflux, 0:self.n], xpostrue, ypostrue, fluxtrue)
                    ## limits
                    setp_imaglimt(gdat, axis)
                    plt.savefig(pathdatartag + '%s_cntpdata%04d.pdf' % (strgtimestmp, jj))
                    
                    # residual map
                    figr, axis = plt.subplots()
                    if gdat.booltimebins:
                        temp = resid[0, :, :] * np.sqrt(weig[0, :, :])
                    else:
                        temp = resid * np.sqrt(weig)
                    axis.imshow(temp, origin='lower', interpolation='none', vmin=-5, vmax=5, cmap=cmapresi)
                    ## overplot point sources
                    supr_catl(axis, self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.stars[gdat.indxflux, 0:self.n], xpostrue, ypostrue, fluxtrue)
                    setp_imaglimt(gdat, axis)
                    plt.savefig(pathdatartag + '%s_cntpresi%04d.pdf' % (strgtimestmp, jj))
                    
                    ## flux histogram
                    figr, axis = plt.subplots()
                    if datatype == 'mock':
                        axis.hist(np.log10(fluxtrue), range=(np.log10(self.trueminf), np.log10(np.max(fluxtrue))), \
                                                                                            log=True, alpha=0.5, label=labldata, histtype=histtype, lw=linewdth, \
                                                                                            facecolor='g', edgecolor='g')
                        axis.hist(np.log10(self.stars[gdat.indxflux, 0:self.n]), range=(np.log10(self.trueminf), np.log10(np.max(fluxtrue))), \
                                                                                            edgecolor='b', facecolor='b', lw=linewdth, \
                                                                                            log=True, alpha=0.5, label='Sample', histtype=histtype)
                    else:
                        colr = 'b'
                        axis.hist(np.log10(self.stars[gdat.indxflux, 0:self.n]), range=(np.log10(self.trueminf), np.ceil(np.log10(np.max(self.stars[gdat.indxflux, 0:self.n])))), \
                                                                                             lw=linewdth, facecolor='b', edgecolor='b', log=True, alpha=0.5, \
                                                                                             label='Sample', histtype=histtype)
                    plt.savefig(pathdatartag + '%s_histflux%04d.pdf' % (strgtimestmp, jj))

            return self.n, self.ng, chi2
    
        
        def idx_parity_stars(self):
            return idx_parity(self.stars[gdat.indxxpos,:], self.stars[gdat.indxypos,:], self.n, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize)
    
        
        def idx_parity_galaxies(self):
            return idx_parity(self.galaxies[gdat.indxxpos,:], self.galaxies[gdat.indxypos,:], self.ng, self.offsetx, self.offsety, self.parity_x, self.parity_y, regsize)
    
        
        def bounce_off_edges(self, catalogue): # works on both stars and galaxies
            mask = catalogue[gdat.indxxpos,:] < 0
            catalogue[gdat.indxxpos, mask] *= -1
            mask = catalogue[gdat.indxxpos,:] > (gdat.sizeimag[0] - 1)
            catalogue[gdat.indxxpos, mask] *= -1
            catalogue[gdat.indxxpos, mask] += 2*(gdat.sizeimag[0] - 1)
            mask = catalogue[gdat.indxypos,:] < 0
            catalogue[gdat.indxypos, mask] *= -1
            mask = catalogue[gdat.indxypos,:] > (gdat.sizeimag[1] - 1)
            catalogue[gdat.indxypos, mask] *= -1
            catalogue[gdat.indxypos, mask] += 2*(gdat.sizeimag[1] - 1)
            # these are all inplace operations, so no return value
    
        
        def in_bounds(self, catalogue):
            return np.logical_and(np.logical_and(catalogue[gdat.indxxpos,:] > 0, catalogue[gdat.indxxpos,:] < (gdat.sizeimag[0] -1)), \
                    np.logical_and(catalogue[gdat.indxypos,:] > 0, catalogue[gdat.indxypos,:] < gdat.sizeimag[1] - 1))
    
        
        def move_stars(self): 
            idx_move = self.idx_parity_stars()
            nw = idx_move.size
            stars0 = self.stars.take(idx_move, axis=1)
            starsp = np.empty_like(stars0)
            f0 = stars0[gdat.indxflux,:]
    
            lindf = np.float32(60./np.sqrt(25.))#134
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
    
            dpos_rms = np.float32(60./np.sqrt(25.))/(np.maximum(f0, pf))#134
            dpos_rms[dpos_rms < 1e-3] = 1e-3
            dx = np.random.normal(size=nw).astype(np.float32)*dpos_rms
            dy = np.random.normal(size=nw).astype(np.float32)*dpos_rms
            starsp[gdat.indxxpos,:] = stars0[gdat.indxxpos,:] + dx
            starsp[gdat.indxypos,:] = stars0[gdat.indxypos,:] + dy
            starsp[gdat.indxflux,:] = pf
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
                mregx = ((gdat.sizeimag[0] / regsize + 1) + 1) / 2 # assumes that sizeimag are multiples of regsize
                mregy = ((gdat.sizeimag[1] / regsize + 1) + 1) / 2
                starsb = np.empty((gdat.numbparastar, nbd), dtype=np.float32)
                starsb[gdat.indxxpos,:] = (np.random.randint(mregx, size=nbd)*2 + self.parity_x + np.random.uniform(size=nbd))*regsize - self.offsetx
                starsb[gdat.indxypos,:] = (np.random.randint(mregy, size=nbd)*2 + self.parity_y + np.random.uniform(size=nbd))*regsize - self.offsety
                starsb[gdat.indxflux,:] = self.trueminf * np.exp(np.random.exponential(scale=1./(self.truealpha-1.),size=nbd))
    
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
            idx_bright = idx_reg.take(np.flatnonzero(self.stars[gdat.indxflux, :].take(idx_reg) > 2*self.trueminf)) # in region!
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
                if gdat.booltimebins:
                    
                    if gdat.verbtype > 1:
                        print 'stars0'
                        summgene(stars0)
                    x0 = stars0[gdat.indxxpos, :]
                    y0 = stars0[gdat.indxypos, :]
                    f0 = stars0[gdat.indxflux, :]
                    lcpr0 = stars0[gdat.indxlcpr, :]
                    #x0, y0, f0, lcpr0 = stars0
                else:
                    x0, y0, f0 = stars0
                fminratio = f0 / self.trueminf
                frac = (1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32)
                
                starsp = np.empty_like(stars0)
                starsp[gdat.indxxpos,:] = x0 + ((1-frac)*dx)
                starsp[gdat.indxypos,:] = y0 + ((1-frac)*dy)
                starsp[gdat.indxflux,:] = f0 * frac
                starsb = np.empty_like(stars0)
                starsb[gdat.indxxpos,:] = x0 - frac*dx
                starsb[gdat.indxypos,:] = y0 - frac*dy
                starsb[gdat.indxflux,:] = f0 * (1-frac)
    
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
                sum_f = stars0[gdat.indxflux,:]
                invpairs = np.empty(nms)
                for k in xrange(nms):
                    xtemp = self.stars[gdat.indxxpos, 0:self.n].copy()
                    ytemp = self.stars[gdat.indxypos, 0:self.n].copy()
                    xtemp[idx_move[k]] = starsp[gdat.indxxpos, k]
                    ytemp[idx_move[k]] = starsp[gdat.indxypos, k]
                    xtemp = np.concatenate([xtemp, starsb[gdat.indxxpos, k:k+1]])
                    ytemp = np.concatenate([ytemp, starsb[gdat.indxypos, k:k+1]])
    
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
                    invpairs[k], idx_kill[k] = neighbours(self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.kickrange, idx_move[k], generate=True)
                    if invpairs[k] > 0:
                        invpairs[k] = 1./invpairs[k]
                    # prevent sources from being involved in multiple proposals
                    if not choosable[idx_kill[k]]:
                        idx_kill[k] = -1
                    if idx_kill[k] != -1:
                        invpairs[k] += 1./neighbours(self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.kickrange, idx_kill[k])
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
    
                f0 = stars0[gdat.indxflux,:]
                fk = starsk[gdat.indxflux,:]
                sum_f = f0 + fk
                fminratio = sum_f / self.trueminf
                frac = f0 / sum_f
                starsp = np.empty_like(stars0)
                starsp[gdat.indxxpos,:] = frac*stars0[gdat.indxxpos,:] + (1-frac)*starsk[gdat.indxxpos,:]
                starsp[gdat.indxypos,:] = frac*stars0[gdat.indxypos,:] + (1-frac)*starsk[gdat.indxypos,:]
                starsp[gdat.indxflux,:] = f0 + fk
                if goodmove:
                    proposal.add_move_stars(idx_move, stars0, starsp)
                    proposal.add_death_stars(idx_kill, starsk)
                # turn bright_n into an array
                bright_n = bright_n - (f0 > 2*self.trueminf) - (fk > 2*self.trueminf) + (starsp[gdat.indxflux,:] > 2*self.trueminf)
            if goodmove:
                factor = np.log(self.truealpha-1) + (self.truealpha-1)*np.log(self.trueminf) - self.truealpha*np.log(frac*(1-frac)*sum_f) + \
                                            np.log(2*np.pi*self.kickrange*self.kickrange) - np.log(gdat.sizeimag[0]*gdat.sizeimag[1]) + np.log(1. - 2./fminratio) + \
                                            np.log(bright_n) + np.log(invpairs) + np.log(sum_f) # last term is Jacobian
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
            f0g = galaxies0[gdat.indxflux,:]
    
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
    
            galaxiesp[gdat.indxxpos,:] = galaxies0[gdat.indxxpos,:] + dxg
            galaxiesp[gdat.indxypos,:] = galaxies0[gdat.indxypos,:] + dyg
            galaxiesp[gdat.indxflux,:] = pfg
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
                mregx = ((gdat.sizeimag[0] / regsize + 1) + 1) / 2 # assumes that sizeimag are multiples of regsize
                mregy = ((gdat.sizeimag[1] / regsize + 1) + 1) / 2
                galaxiesb = np.empty((6,nbd))
                galaxiesb[gdat.indxxpos,:] = (np.random.randint(mregx, size=nbd)*2 + self.parity_x + np.random.uniform(size=nbd))*regsize - self.offsetx
                galaxiesb[gdat.indxypos,:] = (np.random.randint(mregy, size=nbd)*2 + self.parity_y + np.random.uniform(size=nbd))*regsize - self.offsety
                galaxiesb[gdat.indxflux,:] = self.trueminf * np.exp(np.random.exponential(scale=1./(self.truealpha-1.),size=nbd))
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
                    galaxiesb[[gdat.indxxpos, gdat.indxypos, gdat.indxflux],:] = starsk
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
                    starsb = galaxiesk[[gdat.indxxpos, gdat.indxypos, gdat.indxflux],:].copy()
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
            idx_bright = idx_reg_g.take(np.flatnonzero(self.galaxies[gdat.indxflux, :].take(idx_reg_g) > 2*self.trueminf)) # in region and bright enough to make two stars
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
                starsb = np.empty((gdat.numbparastar, nms, 2), dtype=np.float32)
                starsb[gdat.indxxpos, :, [0,1]] = xkg + ((1-frac)*dx), xkg - frac*dx
                starsb[gdat.indxypos, :, [0,1]] = ykg + ((1-frac)*dy), ykg - frac*dy
                starsb[gdat.indxflux, :, [0,1]] = fkg * frac         , fkg * (1-frac)
    
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
                weigoverpairs = np.empty(nms) # w (1/sum w_1 + 1/sum w_2) / 2
                for k in xrange(nms):
                    xtemp = self.stars[gdat.indxxpos, 0:self.n].copy()
                    ytemp = self.stars[gdat.indxypos, 0:self.n].copy()
                    xtemp = np.concatenate([xtemp, starsb[0, k:k+1, gdat.indxxpos], starsb[1, k:k+1, gdat.indxxpos]])
                    ytemp = np.concatenate([ytemp, starsb[0, k:k+1, gdat.indxypos], starsb[1, k:k+1, gdat.indxypos]])
    
                    neighi = neighbours(xtemp, ytemp, self.kickrange_g, self.n)
                    neighj = neighbours(xtemp, ytemp, self.kickrange_g, self.n+1)
                    if neighi > 0 and neighj > 0:
                        weigoverpairs[k] = 1./neighi + 1./neighj
                    else:
                        weigoverpairs[k] = 0.
                weigoverpairs *= 0.5 * np.exp(-dr2/(2.*self.kickrange_g*self.kickrange_g))
                weigoverpairs[weigoverpairs == 0] = 1
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
                    invpairs[k], idx_kill[k,1] = neighbours(self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.kickrange_g, idx_kill[k,0], generate=True)
                    # prevent sources from being involved in multiple proposals
                    if not choosable[idx_kill[k,1]]:
                        idx_kill[k,1] = -1
                    if idx_kill[k,1] != -1:
                        invpairs[k] = 1./invpairs[k]
                        invpairs[k] += 1./neighbours(self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.kickrange_g, idx_kill[k,1])
                        choosable[idx_kill[k,:]] = False
                        nchoosable -= 2
                invpairs *= 0.5
    
                inbounds = (idx_kill[:,1] != -1)
                idx_kill = idx_kill.compress(inbounds, axis=0)
                invpairs = invpairs.compress(inbounds)
                nms = np.sum(inbounds)
                goodmove = nms > 0
    
                starsk = self.stars.take(idx_kill, axis=1) # because stars is (numbparastar, N) and idx_kill is (nms, 2), this is (numbparastar, nms, 2)
                fk = starsk[gdat.indxflux,:,:]
                dx = starsk[gdat.indxxpos,:,1] - starsk[gdat.indxxpos,:,0]
                dy = starsk[gdat.indxxpos,:,1] - starsk[gdat.indxypos,:,0]
                dr2 = dx*dx + dy*dy
                weigoverpairs = np.exp(-dr2/(2.*self.kickrange_g*self.kickrange_g)) * invpairs
                weigoverpairs[weigoverpairs == 0] = 1
                sum_f = np.sum(fk, axis=1)
                fminratio = sum_f / self.trueminf
                frac = fk[:,0] / sum_f
                f1Mf = frac * (1. - frac)
                galaxiesb = np.empty((6, nms))
                galaxiesb[gdat.indxxpos,:] = frac*starsk[gdat.indxxpos,:,0] + (1-frac)*starsk[gdat.indxxpos,:,1]
                galaxiesb[gdat.indxypos,:] = frac*starsk[gdat.indxypos,:,0] + (1-frac)*starsk[gdat.indxypos,:,1]
                galaxiesb[gdat.indxflux,:] = sum_f
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
                    self.truealpha*np.log(f1Mf) - (2*self.truealpha - self.truealpha_g)*np.log(sum_f) - np.log(gdat.sizeimag[0]*gdat.sizeimag[1]) + np.log(1. - 2./fminratio) - \
                    np.log(2*np.pi*self.kickrange_g*self.kickrange_g) + np.log(bright_n/(self.ng+1.-splitsville)) + np.log((self.n-1+2*splitsville)*weigoverpairs) + \
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
            idx_bright = idx_reg_g.take(np.flatnonzero(self.galaxies[gdat.indxflux, :].take(idx_reg_g) > self.trueminf + self.trueminf_g)) # in region and bright enough to make s+g
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
                galaxiesp[gdat.indxxpos,:] = x0g + ((1-frac)*dx)
                galaxiesp[gdat.indxypos,:] = y0g + ((1-frac)*dy)
                galaxiesp[gdat.indxflux,:] = f0g * frac
                pxxg = (xx0g - frac*(1-frac)*dx*dx)/frac
                pxyg = (xy0g - frac*(1-frac)*dx*dy)/frac
                pyyg = (yy0g - frac*(1-frac)*dy*dy)/frac
                galaxiesp[self._XX,:] = pxxg
                galaxiesp[self._XY,:] = pxyg
                galaxiesp[self._YY,:] = pyyg
                starsb = np.empty((gdat.numbparastar, nms), dtype=np.float32)
                starsb[gdat.indxxpos,:] = x0g - frac*dx
                starsb[gdat.indxypos,:] = y0g - frac*dy
                starsb[gdat.indxflux,:] = f0g * (1-frac)
    
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
                    xtemp = self.stars[gdat.indxxpos, 0:self.n].copy()
                    ytemp = self.stars[gdat.indxypos, 0:self.n].copy()
                    xtemp = np.concatenate([xtemp, galaxiesp[gdat.indxxpos, k:k+1], starsb[gdat.indxxpos, k:k+1]])
                    ytemp = np.concatenate([ytemp, galaxiesp[gdat.indxypos, k:k+1], starsb[gdat.indxypos, k:k+1]])
    
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
                    invpairs[k], idx_kill[k] = neighbours(np.concatenate([self.stars[gdat.indxxpos, 0:self.n], self.galaxies[gdat.indxxpos, l:l+1]]), \
                            np.concatenate([self.stars[gdat.indxypos, 0:self.n], self.galaxies[gdat.indxypos, l:l+1]]), self.kickrange_g, self.n, generate=True)
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
                sum_f = galaxies0[gdat.indxflux,:] + starsk[gdat.indxflux,:]
                frac = galaxies0[gdat.indxflux,:] / sum_f
                dx = galaxies0[gdat.indxxpos,:] - starsk[gdat.indxxpos,:]
                dy = galaxies0[gdat.indxypos,:] - starsk[gdat.indxypos,:]
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
                    galaxiesp[gdat.indxxpos,:] = frac*galaxies0[gdat.indxxpos,:] + (1-frac)*starsk[gdat.indxxpos,:]
                    galaxiesp[gdat.indxypos,:] = frac*galaxies0[gdat.indxypos,:] + (1-frac)*starsk[gdat.indxypos,:]
                    galaxiesp[gdat.indxflux,:] = galaxies0[gdat.indxflux,:] + starsk[gdat.indxflux,:]
                    galaxiesp[self._XX,:] = pxxg
                    galaxiesp[self._XY,:] = pxyg
                    galaxiesp[self._YY,:] = pyyg
                    proposal.add_move_galaxies(idx_move_g, galaxies0, galaxiesp)
                    proposal.add_death_stars(idx_kill, starsk)
                # this proposal makes a galaxy that is bright enough to split
                bright_n = bright_n + 1
            if goodmove:
                factor = np.log(self.truealpha-1) - (self.truealpha-1)*np.log(sum_f/self.trueminf) - self.truealpha_g*np.log(frac) - self.truealpha*np.log(1-frac) + \
                        np.log(2*np.pi*self.kickrange_g*self.kickrange_g) - np.log(gdat.sizeimag[0]*gdat.sizeimag[1]) + np.log(1. - (self.trueminf+self.trueminf_g)/sum_f) + \
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
            idx_bright = idx_reg.take(np.flatnonzero(self.galaxies[gdat.indxflux, :].take(idx_reg) > 2*self.trueminf_g)) # in region!
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
                fminratio = galaxies0[gdat.indxflux,:] / self.trueminf_g
                frac = (1./fminratio + np.random.uniform(size=nms)*(1. - 2./fminratio)).astype(np.float32)
                frac_xx = np.random.uniform(size=nms).astype(np.float32)
                frac_xy = np.random.uniform(size=nms).astype(np.float32)
                frac_yy = np.random.uniform(size=nms).astype(np.float32)
                xx_p = galaxies0[self._XX,:] - frac*(1-frac)*dx*dx# moments of just galaxy pair
                xy_p = galaxies0[self._XY,:] - frac*(1-frac)*dx*dy
                yy_p = galaxies0[self._YY,:] - frac*(1-frac)*dy*dy
                galaxiesp = np.empty_like(galaxies0)
                galaxiesp[gdat.indxxpos,:] = galaxies0[gdat.indxxpos,:] + ((1-frac)*dx)
                galaxiesp[gdat.indxypos,:] = galaxies0[gdat.indxypos,:] + ((1-frac)*dy)
                galaxiesp[gdat.indxflux,:] = galaxies0[gdat.indxflux,:] * frac
                galaxiesp[self._XX,:] = xx_p * frac_xx / frac
                galaxiesp[self._XY,:] = xy_p * frac_xy / frac
                galaxiesp[self._YY,:] = yy_p * frac_yy / frac
                galaxiesb = np.empty_like(galaxies0)
                galaxiesb[gdat.indxxpos,:] = galaxies0[gdat.indxxpos,:] - frac*dx
                galaxiesb[gdat.indxypos,:] = galaxies0[gdat.indxypos,:] - frac*dy
                galaxiesb[gdat.indxflux,:] = galaxies0[gdat.indxflux,:] * (1-frac)
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
                sum_f = galaxies0[gdat.indxflux,:]
                invpairs = np.empty(nms)
                for k in xrange(nms):
                    xtemp = self.galaxies[gdat.indxxpos, 0:self.ng].copy()
                    ytemp = self.galaxies[gdat.indxypos, 0:self.ng].copy()
                    xtemp[idx_move_g[k]] = galaxiesp[gdat.indxxpos, k]
                    ytemp[idx_move_g[k]] = galaxiesp[gdat.indxypos, k]
                    xtemp = np.concatenate([xtemp, galaxiesb[gdat.indxxpos, k:k+1]])
                    ytemp = np.concatenate([ytemp, galaxiesb[gdat.indxypos, k:k+1]])
    
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
                    invpairs[k], idx_kill_g[k] = neighbours(self.galaxies[gdat.indxxpos, 0:self.ng], self.galaxies[gdat.indxypos, 0:self.ng], self.kickrange_g, idx_move_g[k], generate=True)
                    if invpairs[k] > 0:
                        invpairs[k] = 1./invpairs[k]
                    # prevent sources from being involved in multiple proposals
                    if not choosable[idx_kill_g[k]]:
                        idx_kill_g[k] = -1
                    if idx_kill_g[k] != -1:
                        invpairs[k] += 1./neighbours(self.galaxies[gdat.indxxpos, 0:self.ng], self.galaxies[gdat.indxypos, 0:self.ng], self.kickrange_g, idx_kill_g[k])
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
                sum_f = galaxies0[gdat.indxflux,:] + galaxiesk[gdat.indxflux,:]
                fminratio = sum_f / self.trueminf_g
                frac = galaxies0[gdat.indxflux,:] / sum_f
    
                galaxiesp = np.empty_like(galaxies0)
                galaxiesp[gdat.indxxpos,:] = frac*galaxies0[gdat.indxxpos,:] + (1-frac)*galaxiesk[gdat.indxxpos,:]
                galaxiesp[gdat.indxypos,:] = frac*galaxies0[gdat.indxypos,:] + (1-frac)*galaxiesk[gdat.indxypos,:]
                galaxiesp[gdat.indxflux,:] = galaxies0[gdat.indxflux,:] + galaxiesk[gdat.indxflux,:]
                dx = galaxies0[gdat.indxxpos,:] - galaxiesk[gdat.indxxpos,:]
                dy = galaxies0[gdat.indxypos,:] - galaxiesk[gdat.indxypos,:]
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
                bright_n = bright_n - (galaxies0[gdat.indxflux,:] > 2*self.trueminf_g) - (galaxiesk[gdat.indxflux,:] > 2*self.trueminf_g) + \
                                                                                                                (galaxiesp[gdat.indxflux,:] > 2*self.trueminf_g)
            if goodmove:
                factor = np.log(self.truealpha_g-1) + (self.truealpha_g-1)*np.log(self.trueminf) - self.truealpha_g*np.log(frac*(1-frac)*sum_f) + \
                    np.log(2*np.pi*self.kickrange_g*self.kickrange_g) - np.log(gdat.sizeimag[0]*gdat.sizeimag[1]) + np.log(1. - 2./fminratio) + np.log(bright_n) + np.log(invpairs) + \
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

    if datatype == 'mock':
        if strgmodl == 'star':
            truth = np.loadtxt(pathliondata + strgdata+'_tru.txt')
            xpostrue = truth[:,0]
            ypostrue = truth[:,1]
            fluxtrue = truth[:,2]
        if strgmodl == 'galx':
            truth_s = np.loadtxt(pathliondata + strgdata+'_str.txt')
            xpostrue = truth_s[:,0]
            ypostrue = truth_s[:,1]
            fluxtrue = truth_s[:,2]
            truth_g = np.loadtxt(pathliondata + strgdata+'_gal.txt')
            truexg = truth_g[:,0]
            trueyg = truth_g[:,1]
            truefg = truth_g[:,2]
            truexxg= truth_g[:,3]
            truexyg= truth_g[:,4]
            trueyyg= truth_g[:,5]
            truerng, theta, phi = from_moments(truexxg, truexyg, trueyyg)
        if strgmodl == 'stargalx':
            truth = np.loadtxt(pathliondata + 'truecnts.txt')
            filetrue = h5py.File(pathliondata + 'true.h5', 'r')
            for attr in filetrue:
                gdat[attr] = filetrue[attr][()]
            filetrue.close()
        
        labldata = 'True'
    else:
        labldata = 'HST 606W'
    
    ntemps = 1
    temps = np.sqrt(2) ** np.arange(ntemps)
    
    regsize = 20
    print 'gdat.sizeimag'
    print gdat.sizeimag
    print ''
    assert gdat.sizeimag[0] % regsize == 0
    assert gdat.sizeimag[1] % regsize == 0
    margin = 10
    
    nstar = Model.nstar
    nstrsamp = np.zeros(numbsamp, dtype=np.int32)
    xpossamp = np.zeros((numbsamp, nstar), dtype=np.float32)
    ypossamp = np.zeros((numbsamp, nstar), dtype=np.float32)
    fluxsamp = np.zeros((numbsamp, nstar), dtype=np.float32)
    
    if gdat.booltimebins:
        lcprsamp = np.zeros((numbsamp, numblcpr, nstar), dtype=np.float32)
    ngsample = np.zeros(numbsamp, dtype=np.int32)
    xgsample = np.zeros((numbsamp, nstar), dtype=np.float32)
    ygsample = np.zeros((numbsamp, nstar), dtype=np.float32)
    fgsample = np.zeros((numbsamp, nstar), dtype=np.float32)
    xxgsample = np.zeros((numbsamp, nstar), dtype=np.float32)
    xygsample = np.zeros((numbsamp, nstar), dtype=np.float32)
    yygsample = np.zeros((numbsamp, nstar), dtype=np.float32)
    
    # construct model for each temperature
    models = [Model() for k in xrange(ntemps)]
    
    # write the chain
    ## h5 file path
    pathh5py = pathdatartag + strgtimestmp + '_chan.h5'
    ## numpy object file path
    pathnump = pathdatartag + strgtimestmp + '_chan.npz'
    
    filearry = h5py.File(pathh5py, 'w')
    print 'Will write the chain to %s...' % pathh5py
    
    if boolplot:
        plt.figure(figsize=(21, 7))
    
    for j in xrange(numbsamp):
        chi2_all = np.zeros(ntemps)
    
        #temptemp = max(50 - 0.1*j, 1)
        temptemp = 1.
        for k in xrange(ntemps):
            _, _, chi2_all[k] = models[k].run_sampler(gdat, temptemp, j, boolplotshow=(k==0)*boolplotshow)
    
        for k in xrange(ntemps-1, 0, -1):
            logfac = (chi2_all[k-1] - chi2_all[k]) * (1./temps[k-1] - 1./temps[k]) / 2.
            if np.log(np.random.uniform()) < logfac:
                print 'swapped', k-1, k
                models[k-1], models[k] = models[k], models[k-1]
    
        nstrsamp[j] = models[0].n
        xpossamp[j,:] = models[0].stars[gdat.indxxpos, :]
        ypossamp[j,:] = models[0].stars[gdat.indxypos, :]
        fluxsamp[j,:] = models[0].stars[gdat.indxflux, :]
    
        if gdat.booltimebins:
            lcprsamp[j, :, :] = models[0].stars[gdat.indxlcpr, :]
    
        if strgmodl == 'galx':
            ngsample[j] = models[0].ng
            xgsample[j,:] = models[0].galaxies[gdat.indxxpos, :]
            ygsample[j,:] = models[0].galaxies[gdat.indxypos, :]
            fgsample[j,:] = models[0].galaxies[gdat.indxflux, :]
            xxgsample[j,:] = models[0].galaxies[Model._XX, :]
            xygsample[j,:] = models[0].galaxies[Model._XY, :]
            yygsample[j,:] = models[0].galaxies[Model._YY, :]
    
    filearry.create_dataset('xpos', data=xpossamp[numbsampburn:, :])
    filearry.create_dataset('ypos', data=ypossamp[numbsampburn:, :])
    filearry.create_dataset('flux', data=fluxsamp[numbsampburn:, :])
    filearry.close()

    path = pathdatartag + 'gdat.p'
    filepick = open(path, 'wb')
    print 'Writing to %s...' % path
    cPickle.dump(gdat, filepick, protocol=cPickle.HIGHEST_PROTOCOL)
    filepick.close()
 
    if boolplotsave:
        print 'Making the animation...'
        cmnd = 'convert -delay 20 -density 200x200 %s/%s_cntpdata*.pdf %s/%s_cntpdata.gif' % (pathdatartag, strgtimestmp, pathdatartag, strgtimestmp)
        print cmnd
        os.system(cmnd)
        cmnd = 'convert -delay 20 -density 200x200 %s/%s_cntpresi*.pdf %s/%s_cntpresi.gif' % (pathdatartag, strgtimestmp, pathdatartag, strgtimestmp)
        print cmnd
        os.system(cmnd)
    
    print 'Saving the numpy object to %s...' % pathnump
    np.savez(pathnump, n=nstrsamp, x=xpossamp, y=ypossamp, f=fluxsamp, ng=ngsample, xg=xgsample, yg=ygsample, fg=fgsample, xxg=xxgsample, xyg=xygsample, yyg=yygsample)
    
    # calculate the condensed catalog
    catlcond = retr_catlcond(rtag)
    
    if boolplot:
        
        # plot the condensed catalog
        if boolplotsave:
            figr, axis = plt.subplots()
            axis.imshow(data, origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data), vmax=np.percentile(data, 95))
            supr_catl(axis, catlcond[:, 0], catlcond[:, 2], catlcond[:, 4], xpostrue, ypostrue, fluxtrue)
            plt.savefig(pathdatartag + '%s_condcatl.pdf' % strgtimestmp)


def retr_catlseed(rtag):
    
    strgtimestmp = rtag[:15]
    
    pathlion = os.environ['LION_PATH'] + '/'
    pathliondata = os.environ["LION_PATH"] + '/Data/'
    pathdata = os.environ['LION_DATA_PATH'] + '/'
    
    pathdatartag = pathdata + rtag + '/'
    os.system('mkdir -p %s' % pathdatartag)

    # maximum number of sources
    maxmnumbsour = 2000
    
    # number of samples used in the seed catalog determination
    numbsampseed = 10
    pathchan = pathdatartag + strgtimestmp + '_chan.h5'
    filechan = h5py.File(pathchan, 'r')
    xpossamp = filechan['xpos'][()][:numbsampseed, :] 
    ypossamp = filechan['ypos'][()][:numbsampseed, :]
    fluxsamp = filechan['flux'][()][:numbsampseed, :]
    PCi, junk = np.mgrid[:numbsampseed, :maxmnumbsour]
    
    mask = fluxsamp > 0
    PCc_all = np.zeros((np.sum(mask), 2))
    PCc_all[:, 0] = xpossamp[mask].flatten()
    PCc_all[:, 1] = ypossamp[mask].flatten()
    PCi = PCi[mask].flatten()
    
    #pos = {}
    #weig = {}
    #for i in xrange(np.sum(mask)):
    # pos[i] = (PCc_all[i, 0], PCc_all[i,1])
    # weig[i] = 0.5
    
    #print pos[0]
    #print PCc_all[0, :]
    #print "graph..."
    #G = nx.read_gpickle('graph')
    #G = nx.geographical_threshold_graph(np.sum(mask), 1./0.75, alpha=1., dim=2., pos=pos, weig=weig)
    
    kdtree = scipy.spatial.KDTree(PCc_all)
    matches = kdtree.query_ball_tree(kdtree, 0.75)
    
    G = nx.Graph()
    G.add_nodes_from(xrange(0, PCc_all.shape[0]))
    
    for i in xrange(PCc_all.shape[0]):
     for j in matches[i]:
      if PCi[i] != PCi[j]:
       G.add_edge(i, j)
    
    current_catalogue = 0
    for i in xrange(PCc_all.shape[0]):
        matches[i].sort()
        bincount = np.bincount(PCi[matches[i]]).astype(np.int)
        ending = np.cumsum(bincount).astype(np.int)
        starting = np.zeros(bincount.size).astype(np.int)
        starting[1:bincount.size] = ending[0:bincount.size-1]
        for j in xrange(bincount.size):
            if j == PCi[i]: # do not match to same catalogue
                continue
            if bincount[j] == 0: # no match to catalogue j
                continue
            if bincount[j] == 1: # exactly one match to catalogue j
                continue
            if bincount[j] > 1:
                dist2 = 0.75**2
                l = -1
                for k in xrange(starting[j], ending[j]):
                    m = matches[i][k]
                    newdist2 = np.sum((PCc_all[i,:] - PCc_all[m,:])**2)
                    if newdist2 < dist2:
                        l = m
                        dist2 = newdist2
                if l == -1:
                    print "didn't find edge even though mutiple matches from this catalogue?"
                for k in xrange(starting[j], ending[j]):
                    m = matches[i][k]
                    if m != l:
                        if G.has_edge(i, m):
                            G.remove_edge(i, m)
                            #print "killed", i, m
    
    catlseed = []
    
    while nx.number_of_nodes(G) > 0:
       deg = nx.degree(G)
       i = max(deg, key=deg.get)
       neighbors = nx.all_neighbors(G, i)
       catlseed.append([PCc_all[i, 0], PCc_all[i, 1], deg[i]])
       G.remove_node(i)
       G.remove_nodes_from(neighbors)
    
    catlseed = np.array(catlseed)

    np.savetxt(pathdatartag + strgtimestmp + '_seed.txt', catlseed)


def retr_catlcond(rtag):

    strgtimestmp = rtag[:15]
    
    # paths
    pathlion = os.environ['LION_PATH'] + '/'
    pathliondata = os.environ["LION_PATH"] + '/Data/'
    pathdata = os.environ['LION_DATA_PATH'] + '/'
    
    pathdatartag = pathdata + rtag + '/'
    os.system('mkdir -p %s' % pathdatartag)

    pathcatlcond = pathdatartag + strgtimestmp + '_catlcond.txt'
    
    # search radius
    radisrch = 0.75
    
    # confidence cut
    cut = 0. 
    
    # gain
    gain = 0.00546689
    
    # read the chain
    print 'Reading the chain...'    
    pathchan = pathdatartag + strgtimestmp + '_chan.h5'
    filechan = h5py.File(pathchan, 'r')
    catlxpos = filechan['xpos'][()] 
    catlypos = filechan['ypos'][()]
    catlflux = filechan['flux'][()]
    
    path = pathdatartag + 'gdat.p'
    filepick = open(path, 'rb')
    print 'Reading %s...' % path
    gdat = cPickle.load(filepick)
    filepick.close()

    numbsamp = len(catlxpos)
    catlnumb = np.zeros(numbsamp, dtype=int)
    indxsamp = np.arange(numbsamp)
    for k in indxsamp:
        catlnumb[k] = len(catlxpos[k])
    filechan.close()
    
    maxmnumbsour = catlxpos.shape[1]
    
    # sort the catalog in decreasing flux
    catlsort = np.zeros((numbsamp, maxmnumbsour, gdat.numbparastar))
    for i in range(0, numbsamp):
        catl = np.zeros((maxmnumbsour, gdat.numbparastar))
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

    retr_catlseed(rtag)
    
    # seed catalog
    ## load the catalog
    pathcatlseed = pathdatartag + strgtimestmp + '_catlseed.txt'
    data = np.loadtxt(pathcatlseed)
    
    ## perform confidence cut
    seedxpos = data[:,0][data[:,2] >= cut*300]
    seedypos = data[:,1][data[:,2] >= cut*300]
    seednumb = data[:,2][data[:,2] >= cut*300]
    
    assert seedxpos.size == seedypos.size
    assert seedxpos.size == seednumb.size
    
    catlseed = np.zeros((seedxpos.size, 2))
    catlseed[:,0] = seedxpos
    catlseed[:,1] = seedypos
    numbsourseed = seedxpos.size
    
    #creates tree, where tree is Pcc_stack
    tree = scipy.spatial.KDTree(PCc_stack)
    
    #keeps track of the clusters
    clusters = np.zeros((numbsamp, len(catlseed) * gdat.numbparastar))
    
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
        for i in range(numbsamp):
    
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
    xposmean = np.zeros(numbsourseed)
    yposmean = np.zeros(numbsourseed)
    fluxmean = np.zeros(numbsourseed)
    magtmean = np.zeros(numbsourseed)
    stdvxpos = np.zeros(numbsourseed)
    stdvypos = np.zeros(numbsourseed)
    stdvflux = np.zeros(numbsourseed)
    stdvmagt = np.zeros(numbsourseed)
    conf = np.zeros(numbsourseed)
    
    # confidence interval defined for err_(x,y,f)
    hi = 84
    lo = 16
    for i in range(0, len(catlseed)):
        x = clusters[:,i][np.nonzero(clusters[:,i])]
        y = clusters[:,i+numbsourseed][np.nonzero(clusters[:,i+numbsourseed])]
        f = clusters[:,i+2*numbsourseed][np.nonzero(clusters[:,i+2*numbsourseed])]
        assert x.size == y.size
        assert x.size == f.size
        conf[i] = x.size/300.0
        xposmean[i] = np.mean(x)
        yposmean[i] = np.mean(y)
        fluxmean[i] = np.mean(f)
        magtmean[i] = 22.5 - 2.5*np.log10(np.mean(f)*gain)
        if x.size > 1:
            stdvxpos[i] = np.percentile(x, hi) - np.percentile(x, lo)
            stdvypos[i] = np.percentile(y, hi) - np.percentile(y, lo)
            stdvflux[i] = np.percentile(f, hi) - np.percentile(f, lo)
            stdvmagt[i] = np.absolute((22.5 - 2.5 * np.log10(np.percentile(f, hi) * gain)) - (22.5 - 2.5 * np.log10(np.percentile(f, lo) * gain)))
        pass
    catlcond = np.zeros((numbsourseed, 9))
    catlcond[:, 0] = xposmean
    catlcond[:, 1] = stdvxpos
    catlcond[:, 2] = yposmean
    catlcond[:, 3] = stdvypos
    catlcond[:, 4] = fluxmean
    catlcond[:, 5] = stdvflux
    catlcond[:, 6] = magtmean
    catlcond[:, 7] = stdvmagt
    catlcond[:, 8] = conf
    
    # save catalog
    np.savetxt(pathcatlcond, catlcond)
    
    return catlcond


# configurations

def cnfg_oldd():

    main( \
         numbsamp=50, \
         colrstyl='lion', \
         boolplotshow=True, \
         boolplotsave=False, \
         #booltimebins=True, \
        )


def cnfg_defa():

    main( \
         numbsamp=3, \
         colrstyl='pcat', \
         #boolplotsave=False, \
         boolplotshow=False, \
         boolplotsave=True, \
         testpsfn=True, \
         #booltimebins=True, \
        )


def cnfg_test():

    main( \
         numbsamp=4, \
         numbloop=10, \
         colrstyl='pcat', \
         #boolplotsave=False, \
         boolplotsave=True, \
         boolplotshow=False, \
         booltimebins=True, \
         verbtype=2, \
         #testpsfn=True, \
         #boolplotsave=True, \
         #boolplotshow=True, \
        )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    else:
        globals().get(sys.argv[1])()





