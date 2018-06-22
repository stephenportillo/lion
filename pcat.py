#import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double
import h5py, datetime
import matplotlib 
import seaborn as sns
sns.set(context='poster', style='ticks', color_codes=True)

import matplotlib.pyplot as plt

import cPickle

import scipy.spatial
import networkx as nx

import time
import astropy.wcs
import astropy.io.fits

import sys, os, warnings

from galaxy import to_moments, from_moments, retr_sers, retr_tranphon

from __init__ import *

class gdatstrt(object):

    def __init__(self):
        self.boollockmodi = False
        pass
    
    def __setattr__(self, attr, valu):
        super(gdatstrt, self).__setattr__(attr, valu)


def eval_modl(gdat, x, y, f, back, numbsidepsfn, coefspix, lcpreval, \
                                            sizeregi=None, margin=0, offsetx=0, offsety=0, weig=None, ref=None, lib=None, sizeimag=None):
    
    assert x.dtype == np.float32
    assert y.dtype == np.float32
    assert f.dtype == np.float32
    assert coefspix.dtype == np.float32
    if ref is not None:
        assert ref.dtype == np.float32
    
    if sizeimag == None:
        sizeimag = gdat.sizeimag

    numbpixlpsfn = numbsidepsfn**2

    numbparaspix = coefspix.shape[0]

    if weig is None:
        if gdat.booltimebins:
            weig = np.full([gdat.numbtime] + sizeimag, 1., dtype=np.float32)
        else:
            weig = np.full(sizeimag, 1., dtype=np.float32)
    if sizeregi is None:
        sizeregi = max(sizeimag[0], sizeimag[1])

    # temp -- sometimes phonions are outside image... what is best way to handle?
    goodsrc = (x > 0) * (x < sizeimag[0] - 1) * (y > 0) * (y < sizeimag[1] - 1)
    x = x.compress(goodsrc)
    y = y.compress(goodsrc)
    f = f.compress(goodsrc)
    if gdat.booltimebins:
        lcpreval = lcpreval.compress(goodsrc, axis=1)

    numbphon = x.size
    rad = numbsidepsfn / 2 # 12 for numbsidepsfn = 25

    numbregiyaxi = sizeimag[1] / sizeregi + 1 # assumes sizeimag % sizeregi = 0?
    numbregixaxi = sizeimag[0] / sizeregi + 1

    ix = np.ceil(x).astype(np.int32)
    dx = ix - x
    iy = np.ceil(y).astype(np.int32)
    dy = iy - y

    dd = np.column_stack((np.full(numbphon, 1., dtype=np.float32), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, dx*dx*dy, dx*dy*dy, dy*dy*dy)).astype(np.float32) * f[:, None]

    if lib is None:
        
        modl = np.full((sizeimag[1]+2*rad+1,sizeimag[0]+2*rad+1), back, dtype=np.float32)
        
        if gdat.verbtype > 1:
            print 'dd'
            summgene(dd)
            print 'coefspix'
            summgene(coefspix)
            print 'numbphon'
            print numbphon
            print 'numbsidepsfn'
            print numbsidepsfn
        recon2 = np.dot(dd, coefspix).reshape((numbphon,numbsidepsfn,numbsidepsfn))
        recon = np.zeros((numbphon,numbsidepsfn,numbsidepsfn), dtype=np.float32)
        recon[:,:,:] = recon2[:,:,:]
        for i in xrange(numbphon):
            modl[iy[i]:iy[i]+rad+rad+1,ix[i]:ix[i]+rad+rad+1] += recon[i,:,:]

        modl = modl[rad:sizeimag[1]+rad,rad:sizeimag[0]+rad]

        if ref is not None:
            diff = ref - modl
        diff2 = np.zeros((numbregiyaxi, numbregixaxi), dtype=np.float64)
        for i in xrange(numbregiyaxi):
            y0 = max(i*sizeregi - offsety - margin, 0)
            y1 = min((i+1)*sizeregi - offsety + margin, sizeimag[1])
            for j in xrange(numbregixaxi):
                x0 = max(j*sizeregi - offsetx - margin, 0)
                x1 = min((j+1)*sizeregi - offsetx + margin, sizeimag[0])
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

        lib(sizeimag[0], sizeimag[1], numbphon, numbsidepsfn, numbparaspix, dd, coefspix, recon, ix, iy, modl, \
                                                reftemp, weig, diff2, sizeregi, margin, offsetx, offsety, gdat.numbtime, gdat.booltimebins, lcpreval)

    if ref is not None:
        return modl, diff2
    else:
        return modl


def retr_dtre(spec):
    
    # temp
    return spec


def psf_poly_fit(gdat, psfnusam, factusam):
    
    assert psfnusam.shape[0] == psfnusam.shape[1] # assert PSF is square
    
    # number of pixels along the side of the upsampled PSF
    numbsidepsfnusam = psfnusam.shape[0]

    # pad by one row and one column
    psfnusampadd = np.zeros((numbsidepsfnusam+1, numbsidepsfnusam+1), dtype=np.float32)
    psfnusampadd[0:numbsidepsfnusam, 0:numbsidepsfnusam] = psfnusam

    # make design matrix for each factusam x factusam region
    numbsidepsfn = numbsidepsfnusam / factusam # dimension of original psf
    nx = factusam + 1
    y, x = np.mgrid[0:nx, 0:nx] / np.float32(factusam)
    x = x.flatten()
    y = y.flatten()
    A = np.column_stack([np.full(nx*nx, 1, dtype=np.float32), x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y]).astype(np.float32)
    
    # number of subpixel parameters
    numbparaspix = A.shape[1]

    # output array of coefficients
    coefspix = np.zeros((numbparaspix, numbsidepsfn, numbsidepsfn), dtype=np.float32)

    # loop over original psf pixels and get fit coefficients
    for i in xrange(numbsidepsfn):
        for j in xrange(numbsidepsfn):
            # solve p = A coefspix for coefspix
            p = psfnusampadd[i*factusam:(i+1)*factusam+1, j*factusam:(j+1)*factusam+1].flatten()
            coefspix[:, i, j] = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, p)) 
    coefspix = coefspix.reshape(coefspix.shape[0], coefspix.shape[1] * coefspix.shape[2])
    
    if gdat.boolplotsave:
        figr, axis = plt.subplots()
        axis.imshow(psfnusampadd, interpolation='none')
        plt.savefig(gdat.pathdatartag + '%s_psfnusampadd.' % gdat.rtag + gdat.strgplotfile)

    return coefspix
   

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


def get_region(x, offsetx, sizeregi):
    return np.floor(x + offsetx).astype(np.int) / sizeregi


# visualization-related functions
def setp_imaglimt(gdat, axis):
    
    axis.set_xlim(-0.5, gdat.sizeimag[0] - 0.5)
    axis.set_ylim(-0.5, gdat.sizeimag[1] - 0.5)
                    
    
def idx_parity(x, y, n, offsetx, offsety, parity_x, parity_y, sizeregi):
    match_x = (get_region(x[0:n], offsetx, sizeregi) % 2) == parity_x
    match_y = (get_region(y[0:n], offsety, sizeregi) % 2) == parity_y
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


class cntrstrt():
    
    def gets(self):
        return self.cntr

    def incr(self, valu=1):
        temp = self.cntr
        self.cntr += valu
        return temp
    
    def __init__(self):
        self.cntr = 0


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


#def summgene(varb):
#
#    print np.amin(varb)
#    print np.amax(varb)
#    print np.mean(varb)
#    print varb.shape


def main( \
         # string characterizing the type of data
         strgdata='sdss0921', \
        
         # numpy array containing the data counts
         cntpdata=None, \
        
         # scalar bias
         bias=None, \
         
         # scalar gain
         gain=None, \

         # string characterizing the type of PSF
         strgpsfn='sdss0921', \
        
         # numpy array containing the PSF counts
         cntppsfn=None, \
            
         # factor by which the PSF is oversampled
         factusam=None, \

         # a Boolean flag indicating whether the data is time-binned
         booltimebins=False, \

         # data path
         pathdata=None, \

         # number of samples
         numbsamp=100, \
    
         # size of the regions in number of pixels
         sizeregi=20, \
         
         # string indicating type of plot file
         strgplotfile='pdf', \

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
        
         # string indicating the type of model
         strgmode='pcat', \
    
         # catalog for forced photometry
         catlforc=None, \

         # reference catalog
         catlrefr=[], \

         # labels for reference catalogs
         lablrefr=[], \

         # colors for reference catalogs
         colrrefr=[], \

         # a string extension to the run tag
         rtagextn=None, \
       
         # level of verbosity
         verbtype=1, \
         
         # diagnostic mode
         diagmode=False, \

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

    #np.seterr(all='raise')

    # load arguments into the global object
    #gdat.boolplotsave = boolplotsave
    #gdat.booltimebins = booltimebins
    #gdat.verbtype = booltimebins

    # time stamp string
    strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
   
    # run tag
    gdat.rtag = 'pcat_' + strgtimestmp + '_%06d' % gdat.numbsamp
    if gdat.rtagextn != None:
        gdat.rtag += '_' + gdat.rtagextn

    if gdat.numbsampburn == None:
        gdat.numbsampburn = int(0.2 * gdat.numbsamp)

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
    if boolplotshow:
        plt.ion()
   
    # plotting 
    gdat.maxmfluxplot = 1e6
    gdat.minmfluxplot = 1e1
    
    gdat.numbrefr = len(gdat.catlrefr)
    gdat.indxrefr = np.arange(gdat.numbrefr)

    if strgmode == 'pcat':
        probprop = np.array([80., 40., 40.])
            
        if strgmodl == 'galx':
            probprop = np.array([80., 40., 40., 80., 40., 40., 40., 40., 40.])
    else:
        probprop = np.array([80., 0., 0.])
    probprop /= np.sum(probprop)
    
    # ix, iy = 0. to 3.999
    def plot_psfn(gdat, numbsidepsfn, coefspix, psf, ix, iy, lib=None):
        
        lcpreval = np.array([[0.]], dtype=np.float32)
        xpos = np.array([12. - ix / 5.], dtype=np.float32)
        ypos = np.array([12. - iy / 5.], dtype=np.float32)
        flux = np.array([1.], dtype=np.float32)
        back = 0.
        sizeimag = [25, 25]
        psf0 = eval_modl(gdat, xpos, ypos, flux, back, numbsidepsfn, coefspix, lcpreval, lib=lib, sizeimag=sizeimag)
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
            plt.savefig(gdat.pathdatartag + '%s_psfn.' % gdat.rtag + gdat.strgplotfile)
        else:
            plt.show()
    
    def plot_lcur(gdat):
       
        cntr = 0
        for k in gdat.indxsourcond:
            
            errrspec = np.empty((2, gdat.numbtime))
            for i in range(2):
                errrspec[i, 0]  = catlcond[gdat.indxflux, k, i]
                errrspec[i, 1:] = catlcond[gdat.indxflux, k, i] * catlcond[gdat.indxlcpr, k, i]
            
            errrspecdtre = retr_dtre(errrspec)
            medimeanspecdtre = np.median(errrspecdtre[0, :])
            stdvflux = np.sqrt((errrspecdtre[0, :] - medimeanspecdtre)**2)
            
            if k % 10 == 0 and k != gdat.numbsourcond - 1:
                figr, axis = plt.subplots()
            
            axis.errorbar(gdat.indxtime, errrspecdtre[0, :], yerr=errrspecdtre[1, :])
            
            axis.set_title(stdvflux)
            if k % 10 == 0 and k != 0:
                plt.savefig(gdat.pathdatartag + '%s_lcur%04d.' % (gdat.rtag, cntr) + gdat.strgplotfile)
                cntr += 1
    
    pathlion, gdat.pathdata = retr_path(gdat.pathdata)
        
    # check if source has been changed after compilation
    if os.path.getmtime(pathlion + 'blas.c') > os.path.getmtime(pathlion + 'blas.so'):
        warnings.warn('blas.c modified after compiled blas.so', Warning)
    
    gdat.pathdatartag = gdat.pathdata + gdat.rtag + '/'
    os.system('mkdir -p %s' % gdat.pathdatartag)

    # constants
    ## number of bands (i.e., energy bins)
    numbener = 1
    
    boolplot = boolplotshow or boolplotsave

    # parse PSF
    
    numbsidepsfnusam = cntppsfn.shape[0]
    numbsidepsfn = numbsidepsfnusam / factusam
    
    if gdat.verbtype > 1:
        print 'numbsidepsfn'
        print numbsidepsfn
        print 'factusam'
        print factusam
    coefspix = psf_poly_fit(gdat, cntppsfn, factusam)
    npar = coefspix.shape[0]
    
    # construct C library
    array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
    array_3d_float = npct.ndpointer(dtype=np.float32, ndim=3, flags="C_CONTIGUOUS")
    array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    array_2d_double = npct.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
    array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")
    libmmult = npct.load_library(pathlion + 'blas', '.')
    libmmult.pcat_model_eval.restype = None
    
    #void pcat_model_eval(int NX, int NY, int nstar, int numbsidepsfn, int k, float* A, float* B, float* C, int* x,
	#int* y, float* image, float* ref, float* weig, double* diff2, int sizeregi, int margin, int offsetx, int offsety)
    
    #lib(gdat.sizeimag[0], gdat.sizeimag[1], nstar, numbsidepsfn, coefspix.shape[0], dd, coefspix, recon, ix, iy, image, reftemp, weig, diff2, sizeregi, margin, offsetx, offsety)
    
    #libmmult.pcat_model_eval.argtypes = [c_int NX gdat.sizeimag[0], c_int NY gdat.sizeimag[1], c_int nstar, c_int numbsidepsfn, c_int k coefspix.shape[0]
    
    # array_2d_float A dd, array_2d_float B coefspix, array_2d_float C recon
    # array_1d_int x ix, array_1d_int y iy, array_2d_float image, array_2d_float ref reftemp, array_2d_float weig weig, array_2d_double diff2, c_int sizeregi, 
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
    if not isinstance(gdat.cntpdata, np.ndarray):
        raise Exception('')
    
    if gdat.cntpdata.ndim == 3:
        numbsideypos = gdat.cntpdata.shape[1]
        numbsidexpos = gdat.cntpdata.shape[2]
    else:
        numbsideypos = gdat.cntpdata.shape[0]
        numbsidexpos = gdat.cntpdata.shape[1]

    gdat.sizeimag = (numbsidexpos, numbsideypos)
    numbpixl = numbsidexpos * numbsideypos

    if gdat.booltimebins:
        if gdat.cntpdata.ndim == 2:
            raise Exception('')
        gdat.numbtime = gdat.cntpdata.shape[0]
        gdat.numblcpr = gdat.numbtime - 1
        indxtime = np.arange(gdat.numbtime)
    else:
        gdat.numbtime = 1
    gdat.indxtime = np.arange(gdat.numbtime)

    print 'Image width and height: %d %d pixels' % (numbsidexpos, numbsideypos)
    
    gdat.numbdata = numbpixl * gdat.numbtime
    
    # plots
    ## PSF
    if boolplot and testpsfn:
        plot_psfn(gdat, numbsidepsfn, coefspix, cntppsfn, np.float32(np.random.uniform()*4), np.float32(np.random.uniform()*4), libmmult.pcat_model_eval)
    
    stdvproplcpr = 1e-6

    ## data
    if False and gdat.booltimebins:
        for k in gdat.indxtime:
            figr, axis = plt.subplots()
            axis.imshow(data[k, :, :], origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data), vmax=np.percentile(data, 95))
            ## limits
            setp_imaglimt(gdat, axis)
            plt.savefig(gdat.pathdatartag + '%s_cntpdatainit%04d.' % (gdat.rtag, k) + gdat.strgplotfile)
        print 'Making the animation...'
        cmnd = 'convert -delay 20 -density 200x200 %s/%s_cntpdatainit*.%s %s/%s_cntpdatainit.gif' % (gdat.pathdatartag, gdat.rtag, gdat.strgplotfile, gdat.pathdatartag, gdat.rtag)
        print cmnd
        os.system(cmnd)
    
    variance = gdat.cntpdata / gain
    weig = 1. / variance # inverse variance
   
    cntr = cntrstrt()
    gdat.indxxpos = cntr.incr()
    gdat.indxypos = cntr.incr()
    gdat.indxflux = cntr.incr()
    if gdat.booltimebins:
        gdat.indxlcpr = np.zeros(gdat.numblcpr, dtype=int)
        for k in range(gdat.numblcpr):
            gdat.indxlcpr[k] = cntr.incr()
        if gdat.verbtype > 1:
            print 'gdat.indxlcpr'
            print gdat.indxlcpr
            summgene(gdat.indxlcpr)
            print

    if 'galx' in strgmodl:
        _XX = cntr.incr()
        _XY = cntr.incr()
        _YY = cntr.incr()
    
    gdat.numbparastar = cntr.gets()
    gdat.indxconf = cntr.incr()
    gdat.numbparacatlcond = 1 + 2 * gdat.numbparastar
    
    class Proposal:
    
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
            self.lcprphon = np.array([[]], dtype=np.float32)
            
            #self.back = None

        
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
            
            if gdat.booltimebins:
                if self.lcprphon.size == 0:
                    self.lcprphon = np.copy(stars[gdat.indxlcpr, :])
                else:
                    self.lcprphon = np.append(self.lcprphon, stars[gdat.indxlcpr, :], axis=1)
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

    def supr_catl(gdat, axis, xpos, ypos, flux, boolcond=False):
        
        for k in gdat.indxrefr:
            mask = catlrefr[k]['flux'] > 250
            size = sizemrkr * catlrefr[k]['flux'][mask]
            axis.scatter(catlrefr[k]['xpos'][mask], catlrefr[k]['ypos'][mask], marker='+', s=size, color=gdat.colrrefr[k], lw=2, alpha=0.7)
            mask = np.logical_not(mask)
            axis.scatter(catlrefr[k]['xpos'][mask], catlrefr[k]['ypos'][mask], marker='+', s=size, color=colrbrgt, lw=2, alpha=0.7)
        if boolcond:
            colr = 'yellow'
        else:
            colr = 'b'
        axis.scatter(xpos, ypos, marker='x', s=sizemrkr*flux, color=colr, lw=2, alpha=0.7)
    
    
    class Model:
        
        # should these be class or instance variables?
        nstar = 2000
        trueminf = np.float32(250.)#*136)
        truealpha = np.float32(2.00)
    
        ngalx = 100
        trueminf_g = np.float32(250.)#*136)
        truealpha_g = np.float32(2.00)
        truermin_g = np.float32(1.00)
    
        gridphon, amplphon = retr_sers(sersindx=2.)
    
        penalty = 1.5
        penalty_g = 3.0
        kickrange = 1.
        kickrange_g = 1.
        
        def __init__(self):

            # initialize parameters
            self.n = np.random.randint(self.nstar)+1
            self.stars = np.zeros((gdat.numbparastar, self.nstar), dtype=np.float32)
            self.stars[:, :self.n] = np.random.uniform(size=(gdat.numbparastar, self.n))  # refactor into some sort of prior function?
            
            if gdat.strgmode == 'pcat':
                self.stars[gdat.indxxpos, :self.n] *= gdat.sizeimag[0] - 1
                self.stars[gdat.indxypos, :self.n] *= gdat.sizeimag[1] - 1
            else:
                self.stars[gdat.indxxpos, :self.n] = gdat.catlforc['xpos']
                self.stars[gdat.indxypos, :self.n] = gdat.catlforc['ypos']
            self.stars[gdat.indxflux, :self.n] **= -1. / (self.truealpha - 1.)
            self.stars[gdat.indxflux, :self.n] *= self.trueminf
            if gdat.booltimebins:
                self.stars[gdat.indxlcpr, :self.n] = 1e-6 * np.random.randn(self.n * gdat.numblcpr).reshape((gdat.numblcpr, self.n)) + 1.
    
            self.ng = 0
            if strgmodl == 'galx':
                self.ng = np.random.randint(self.ngalx)+1
                self.galaxies = np.zeros((6,self.ngalx), dtype=np.float32)
                # temp -- 3 should be generalized to temporal modeling
                self.galaxies[[gdat.indxxpos,gdat.indxypos,gdat.indxflux],0:self.ng] = np.random.uniform(size=(3, self.ng))
                self.galaxies[gdat.indxxpos, :self.ng] *= gdat.sizeimag[0] - 1
                self.galaxies[gdat.indxypos, :self.ng] *= gdat.sizeimag[1]-1
                self.galaxies[gdat.indxflux, :self.ng] **= -1./(self.truealpha_g - 1.)
                self.galaxies[gdat.indxflux, :self.ng] *= self.trueminf_g
                self.galaxies[[self._XX,self._XY,self._YY],0:self.ng] = self.moments_from_prior(self.truermin_g, self.ng)
            self.back = np.float32(179)

        
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
    
            self.offsetx = np.random.randint(sizeregi)
            self.offsety = np.random.randint(sizeregi)
            self.nregx = gdat.sizeimag[0] / sizeregi + 1
            self.nregy = gdat.sizeimag[1] / sizeregi + 1
    
            resid = gdat.cntpdata.copy() # residual for zero image is data
            lcpreval = np.array([[0.]], dtype=np.float32)
            if strgmodl == 'star':
                xposeval = self.stars[gdat.indxxpos,0:self.n]
                yposeval = self.stars[gdat.indxypos,0:self.n]
                fluxeval = self.stars[gdat.indxflux,0:self.n]
                if gdat.booltimebins:
                    lcpreval = self.stars[gdat.indxlcpr, :self.n]

                    if gdat.diagmode:
                        chec_lcpr(lcpreval)
            else:
                xposphon, yposphon, specphon = retr_tranphon(self.gridphon, self.amplphon, self.galaxies[:,0:self.ng])
                xposeval = np.concatenate([self.stars[gdat.indxxpos,0:self.n], xposphon]).astype(np.float32)
                yposeval = np.concatenate([self.stars[gdat.indxypos,0:self.n], yposphon]).astype(np.float32)
                fluxeval = np.concatenate([self.stars[gdat.indxflux,0:self.n], specphon]).astype(np.float32)
            numbphon = xposeval.size
            model, diff2 = eval_modl(gdat, xposeval, yposeval, fluxeval, self.back, numbsidepsfn, coefspix, lcpreval, \
                                                               weig=weig, ref=resid, lib=libmmult.pcat_model_eval, \
                                                               #weig=weig, ref=resid, lib=None, \
                                                               sizeregi=sizeregi, margin=margin, offsetx=self.offsetx, offsety=self.offsety)
            logL = -0.5*diff2
            resid -= model
            if gdat.verbtype > 1:
                print 'model'
                summgene(model)
                print 'diff2'
                summgene(diff2)
                print 'logL'
                summgene(logL)
                print 'resid'
                summgene(resid)
                print 
            
            if gdat.diagmode:
                if not np.isfinite(resid).all():
                    raise Exception('')

            for i in xrange(gdat.numbloop):

                if gdat.verbtype > 1:
                    print 'Loop %d started.' % i
                
                t1 = time.clock()
                rtype = np.random.choice(probprop.size, p=probprop)
                movetype[i] = rtype
                # defaults
                pn = self.n
                dback = np.float32(0.)

                # should regions be perturbed randomly or systematically?
                self.parity_x = np.random.randint(2)
                self.parity_y = np.random.randint(2)
    
                movetypes = ['P *', 'BD *', 'MS *', 'P g', 'BD g', '*-g', '**-g', '*g-g', 'MS g']
                movefns = [self.move_stars, self.birth_death_stars, self.merge_split_stars, self.move_galaxies, self.birth_death_galaxies, self.star_galaxy, \
                                                                                                self.twostars_galaxy, self.stargalaxy_galaxy, self.merge_split_galaxies]
                
                # make the proposal
                proposal = movefns[rtype]()
                
                if gdat.verbtype > 1:
                    print 'rtype'
                    print rtype
                
                dt1[i] = time.clock() - t1
    
                if proposal.goodmove:
                    t2 = time.clock()
                    dmodel, diff2 = eval_modl(gdat, proposal.xphon, proposal.yphon, proposal.fphon, dback, numbsidepsfn, coefspix, proposal.lcprphon, \
                                weig=weig, ref=resid, lib=libmmult.pcat_model_eval, sizeregi=sizeregi, margin=margin, offsetx=self.offsetx, offsety=self.offsety)
                    plogL = -0.5*diff2
                    dt2[i] = time.clock() - t2
    
                    t3 = time.clock()
                    refx, refy = proposal.get_ref_xy()
                    regionx = get_region(refx, self.offsetx, sizeregi)
                    regiony = get_region(refy, self.offsety, sizeregi)
    
                    ###
                    '''
                    if i == 0:
                        yy, xx = np.mgrid[0:100,0:100]
                        rxx = get_region(xx, self.offsetx, sizeregi) % 2
                        ryy = get_region(yy, self.offsety, sizeregi) % 2
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
                    libmmult.pcat_imag_acpt(gdat.sizeimag[0], gdat.sizeimag[1], dmodel, dmodel_acpt, acceptreg, sizeregi, margin, self.offsetx, self.offsety)
                    # using this dmodel containing only accepted moves, update logL
                    diff2.fill(0)
                    libmmult.pcat_like_eval(gdat.sizeimag[0], gdat.sizeimag[1], dmodel_acpt, resid, weig, diff2, sizeregi, margin, self.offsetx, self.offsety)
                    logL = -0.5*diff2
                    resid -= dmodel_acpt # has to occur after pcat_like_eval, because resid is used as ref
                    model += dmodel_acpt
                    
                    # implement accepted moves
                    if proposal.idx_move is not None:
                        starsp = proposal.starsp.compress(acceptprop, axis=1)
                        idx_move_a = proposal.idx_move.compress(acceptprop)
                        self.stars[:, idx_move_a] = starsp
                        
                        if gdat.verbtype > 1:
                            print 'proposal.starsp'
                            summgene(proposal.starsp)
                        
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
                    if proposal.goodmove:
                        print 'dmodel'
                        summgene(dmodel)
                    print 'accept[i]'
                    print accept[i]
                    print 'outbounds[i]'
                    print outbounds[i]
                    print 'Loop %d ended.' % i
                    print
                    print
                    print

            numbdoff = retr_numbdoff(self.n, self.ng, numbener)
            chi2 = np.sum(weig * (gdat.cntpdata - model) * (gdat.cntpdata - model))
            chi2doff = chi2 / (gdat.numbdata - numbdoff)
            fmtstr = '\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f (Pg) %0.3f (BDg) %0.3f (S-g) %0.3f (gSg) %0.3f (gMS) %0.3f'
            print 'Sample %d' % jj
            print 'Temperature', temperature, 'background', self.back, 'N_star', self.n, 'N_gal', self.ng, 'N_phon', numbphon, \
                                                                                            'chi^2', chi2, 'numbdoff', numbdoff, 'chi2doff', chi2doff
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
                            plt.scatter(catlrefr[k]['xpos'][mask], catlrefr[k]['ypos'][mask], marker='+', s=np.sqrt(fluxtrue[mask]), color='g')
                    if strgmodl == 'galx':
                        plt.scatter(self.galaxies[gdat.indxxpos, 0:self.ng], self.galaxies[gdat.indxypos, 0:self.ng], marker='2', \
                                                                                            s=sizemrkr*self.galaxies[gdat.indxflux, 0:self.ng], color='r')
                        a, theta, phi = from_moments(self.galaxies[self._XX, 0:self.ng], self.galaxies[self._XY, 0:self.ng], self.galaxies[self._YY, 0:self.ng])
                        plt.scatter(self.galaxies[gdat.indxxpos, 0:self.ng], self.galaxies[gdat.indxypos, 0:self.ng], marker='o', s=4*a*a, edgecolors='red', facecolors='none')
                    
                    supr_catl(gdat, plt.gca(), self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.stars[gdat.indxflux, 0:self.n])
                    
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
                                                                                        log=True, alpha=0.5, label=gdat.lablrefr[k], histtype=histtype, lw=linewdth, \
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
                        temp = gdat.cntpdata[0, :, :]
                    else:
                        temp = gdat.cntpdata
                    axis.imshow(temp, origin='lower', interpolation='none', cmap='Greys', vmin=np.min(gdat.cntpdata), vmax=np.percentile(gdat.cntpdata, 95))
                    ## overplot point sources
                    supr_catl(gdat, axis, self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.stars[gdat.indxflux, 0:self.n])
                    ## limits
                    setp_imaglimt(gdat, axis)
                    plt.savefig(gdat.pathdatartag + '%s_cntpdata%04d.' % (gdat.rtag, jj) + gdat.strgplotfile)
                    
                    # residual map
                    figr, axis = plt.subplots()
                    if gdat.booltimebins:
                        temp = resid[0, :, :] * np.sqrt(weig[0, :, :])
                    else:
                        temp = resid * np.sqrt(weig)
                    axis.imshow(temp, origin='lower', interpolation='none', vmin=-5, vmax=5, cmap=cmapresi)
                    ## overplot point sources
                    supr_catl(gdat, axis, self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.stars[gdat.indxflux, 0:self.n])
                    setp_imaglimt(gdat, axis)
                    plt.savefig(gdat.pathdatartag + '%s_cntpresi%04d.' % (gdat.rtag, jj) + gdat.strgplotfile)
                    
                    ## flux histogram
                    figr, axis = plt.subplots()
                    for k in gdat.indxrefr:
                        axis.hist(np.log10(catlrefr[k]['flux']), log=True, alpha=0.5, label=gdat.lablrefr[k], histtype=histtype, lw=linewdth, \
                                                                                                                    facecolor=gdat.colrrefr[k], edgecolor=gdat.colrrefr[k])
                    axis.hist(np.log10(self.stars[gdat.indxflux, 0:self.n]), edgecolor='b', facecolor='b', lw=linewdth, log=True, alpha=0.5, label='Sample', histtype=histtype)
                    axis.set_xlim([gdat.minmfluxplot, gdat.maxmfluxplot])
                    plt.savefig(gdat.pathdatartag + '%s_histflux%04d.' % (gdat.rtag, jj) + gdat.strgplotfile)

            return self.n, self.ng, chi2
    
        
        def idx_parity_stars(self):
            return idx_parity(self.stars[gdat.indxxpos,:], self.stars[gdat.indxypos,:], self.n, self.offsetx, self.offsety, self.parity_x, self.parity_y, sizeregi)
    
        
        def idx_parity_galaxies(self):
            return idx_parity(self.galaxies[gdat.indxxpos,:], self.galaxies[gdat.indxypos,:], self.ng, self.offsetx, self.offsety, self.parity_x, self.parity_y, sizeregi)
    
        
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
            if gdat.booltimebins:
                starsp[gdat.indxlcpr,:] = stars0[gdat.indxlcpr, :] + np.random.normal(size=nw).astype(np.float32) * stdvproplcpr
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
                mregx = ((gdat.sizeimag[0] / sizeregi + 1) + 1) / 2 # assumes that sizeimag are multiples of sizeregi
                mregy = ((gdat.sizeimag[1] / sizeregi + 1) + 1) / 2
                starsb = np.empty((gdat.numbparastar, nbd), dtype=np.float32)
                starsb[gdat.indxxpos,:] = (np.random.randint(mregx, size=nbd)*2 + self.parity_x + np.random.uniform(size=nbd))*sizeregi - self.offsetx
                starsb[gdat.indxypos,:] = (np.random.randint(mregy, size=nbd)*2 + self.parity_y + np.random.uniform(size=nbd))*sizeregi - self.offsety
                starsb[gdat.indxflux,:] = self.trueminf * np.exp(np.random.exponential(scale=1./(self.truealpha-1.),size=nbd))
                if gdat.booltimebins:
                    starsb[gdat.indxlcpr, :] = np.random.randn(nbd) * 1e-6 + 1.
    
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
            
            boolsplt = np.random.randint(2)
            idx_reg = self.idx_parity_stars()
            sum_f = 0
            low_n = 0
            idx_bright = idx_reg.take(np.flatnonzero(self.stars[gdat.indxflux, :].take(idx_reg) > 2*self.trueminf)) # in region!
            numbbrgt = idx_bright.size
    
            numbspmr = (self.nregx * self.nregy) / 4
            goodmove = False
            proposal = Proposal()
            # split
            if boolsplt and self.n > 0 and self.n < self.nstar and numbbrgt > 0: # need something to split, but don't exceed nstar
                numbspmr = min(numbspmr, numbbrgt, self.nstar - self.n) # need bright source AND room for split source
                dx = (np.random.normal(size=numbspmr)*self.kickrange).astype(np.float32)
                dy = (np.random.normal(size=numbspmr)*self.kickrange).astype(np.float32)
                idx_move = np.random.choice(idx_bright, size=numbspmr, replace=False)
                stars0 = self.stars.take(idx_move, axis=1)
                if gdat.verbtype > 1:
                    print 'stars0'
                    summgene(stars0)
                x0 = stars0[gdat.indxxpos, :]
                y0 = stars0[gdat.indxypos, :]
                if gdat.booltimebins:
                    spec = np.empty(gdat.numbtime)
                    spec[0] = stars0[gdat.indxflux, :]
                    spec[1:] = spec[0] * stars0[gdat.indxlcpr, :]
                    f0 = np.mean(spec)
                    lcpr0 = stars0[gdat.indxlcpr, :]
                else:
                    f0 = stars0[gdat.indxflux, :]
                    x0, y0, f0 = stars0
                fminratio = f0 / self.trueminf
                frac = (1./fminratio + np.random.uniform(size=numbspmr)*(1. - 2./fminratio)).astype(np.float32)
                
                starsp = np.empty_like(stars0)
                starsp[gdat.indxxpos,:] = x0 + ((1-frac)*dx)
                starsp[gdat.indxypos,:] = y0 + ((1-frac)*dy)
                if gdat.booltimebins:
                    starsp[gdat.indxlcpr, :] = f0 * frac * (1. + np.random.rand(gdat.numblcpr) * 1e-6) / gdat.numbtime
                    starsp[gdat.indxflux, :] = f0 * frac - sum(starsp[gdat.indxlcpr, :])
                else:
                    starsp[gdat.indxflux,:] = f0 * frac
                starsb = np.empty_like(stars0)
                starsb[gdat.indxxpos,:] = x0 - frac*dx
                starsb[gdat.indxypos,:] = y0 - frac*dy
                if gdat.booltimebins:
                    starsb[gdat.indxlcpr, :] = fluxtotl[1:] - starsp[gdat.indxlcprZZ, :]
                    starsb[gdat.indxflux, :] = f0 * (1. - frac) - sum(starsp[gdat.indxlcpr, :])
                else:
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
                numbspmr = idx_move.size
                goodmove = numbspmr > 0
                if goodmove:
                    proposal.add_move_stars(idx_move, stars0, starsp)
                    proposal.add_birth_stars(starsb)
    
                # need to calculate factor
                sum_f = stars0[gdat.indxflux,:]
                invpairs = np.empty(numbspmr)
                for k in xrange(numbspmr):
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
            elif not boolsplt and idx_reg.size > 1: # need two things to merge!
                numbspmr = min(numbspmr, idx_reg.size/2)
                idx_move = np.empty(numbspmr, dtype=np.int)
                idx_kill = np.empty(numbspmr, dtype=np.int)
                choosable = np.zeros(self.nstar, dtype=np.bool)
                choosable[idx_reg] = True
                nchoosable = float(idx_reg.size)
                invpairs = np.empty(numbspmr)
    
                for k in xrange(numbspmr):
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
                numbspmr = idx_move.size
                goodmove = numbspmr > 0
    
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
                # turn numbbrgt into an array
                numbbrgt = numbbrgt - (f0 > 2*self.trueminf) - (fk > 2*self.trueminf) + (starsp[gdat.indxflux,:] > 2*self.trueminf)
            if goodmove:
                factor = np.log(self.truealpha-1) + (self.truealpha-1)*np.log(self.trueminf) - self.truealpha*np.log(frac*(1-frac)*sum_f) + \
                                            np.log(2*np.pi*self.kickrange*self.kickrange) - np.log(gdat.sizeimag[0]*gdat.sizeimag[1]) + np.log(1. - 2./fminratio) + \
                                            np.log(numbbrgt) + np.log(invpairs) + np.log(sum_f) # last term is Jacobian
                if not boolsplt:
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
                mregx = ((gdat.sizeimag[0] / sizeregi + 1) + 1) / 2 # assumes that sizeimag are multiples of sizeregi
                mregy = ((gdat.sizeimag[1] / sizeregi + 1) + 1) / 2
                galaxiesb = np.empty((6,nbd))
                galaxiesb[gdat.indxxpos,:] = (np.random.randint(mregx, size=nbd)*2 + self.parity_x + np.random.uniform(size=nbd))*sizeregi - self.offsetx
                galaxiesb[gdat.indxypos,:] = (np.random.randint(mregy, size=nbd)*2 + self.parity_y + np.random.uniform(size=nbd))*sizeregi - self.offsety
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
            boolsplt = np.random.randint(2)
            idx_reg = self.idx_parity_stars() # stars
            idx_reg_g = self.idx_parity_galaxies() # galaxies
            sum_f = 0
            low_n = 0
            idx_bright = idx_reg_g.take(np.flatnonzero(self.galaxies[gdat.indxflux, :].take(idx_reg_g) > 2*self.trueminf)) # in region and bright enough to make two stars
            numbbrgt = idx_bright.size # can only split bright galaxies
    
            numbspmr = (self.nregx * self.nregy) / 4
            goodmove = False
            proposal = Proposal()
            # split
            if boolsplt and self.ng > 0 and self.n < self.nstar-2 and numbbrgt > 0: # need something to split, but don't exceed nstar
                numbspmr = min(numbspmr, numbbrgt, (self.nstar-self.n)/2) # need bright galaxy AND room for split stars
                idx_kill_g = np.random.choice(idx_bright, size=numbspmr, replace=False)
                galaxiesk = self.galaxies.take(idx_kill_g, axis=1)
                xkg, ykg, fkg, xxkg, xykg, yykg = galaxiesk
                fminratio = fkg / self.trueminf # again, care about fmin for stars
                frac = (1./fminratio + np.random.uniform(size=numbspmr)*(1. - 2./fminratio)).astype(np.float32)
                f1Mf = frac * (1. - frac) # frac(1 - frac)
                agalx, theta, phi = from_moments(xxkg, xykg, yykg)
                dx = agalx * np.cos(phi) / np.sqrt(2 * f1Mf)
                dy = agalx * np.sin(phi) / np.sqrt(2 * f1Mf)
                dr2 = dx*dx + dy*dy
                starsb = np.empty((gdat.numbparastar, numbspmr, 2), dtype=np.float32)
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
                numbspmr = fkg.size
                goodmove = numbspmr > 0
                if goodmove:
                    proposal.add_death_galaxies(idx_kill_g, galaxiesk)
                    proposal.add_birth_stars(starsb)
    
                # need star pairs to calculate factor
                sum_f = fkg
                weigoverpairs = np.empty(numbspmr) # w (1/sum w_1 + 1/sum w_2) / 2
                for k in xrange(numbspmr):
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
            elif not boolsplt and idx_reg.size > 1: # need two things to merge!
                numbspmr = min(numbspmr, idx_reg.size/2, self.ngalx-self.ng)
                idx_kill = np.empty((numbspmr, 2), dtype=np.int)
                choosable = np.zeros(self.nstar, dtype=np.bool)
                choosable[idx_reg] = True
                nchoosable = float(idx_reg.size)
                invpairs = np.empty(numbspmr)
    
                for k in xrange(numbspmr):
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
                numbspmr = np.sum(inbounds)
                goodmove = numbspmr > 0
    
                starsk = self.stars.take(idx_kill, axis=1) # because stars is (numbparastar, N) and idx_kill is (numbspmr, 2), this is (numbparastar, numbspmr, 2)
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
                galaxiesb = np.empty((6, numbspmr))
                galaxiesb[gdat.indxxpos,:] = frac*starsk[gdat.indxxpos,:,0] + (1-frac)*starsk[gdat.indxxpos,:,1]
                galaxiesb[gdat.indxypos,:] = frac*starsk[gdat.indxypos,:,0] + (1-frac)*starsk[gdat.indxypos,:,1]
                galaxiesb[gdat.indxflux,:] = sum_f
                u = np.random.uniform(low=3e-4, high=1., size=numbspmr).astype(np.float32) #3e-4 for numerics
                theta = np.arccos(u).astype(np.float32)
                galaxiesb[self._XX,:] = f1Mf*(dx*dx+u*u*dy*dy)
                galaxiesb[self._XY,:] = f1Mf*(1-u*u)*dx*dy
                galaxiesb[self._YY,:] = f1Mf*(dy*dy+u*u*dx*dx)
                # this move proposes a splittable galaxy
                numbbrgt += 1
                if goodmove:
                    proposal.add_death_stars(idx_kill, starsk)
                    proposal.add_birth_galaxies(galaxiesb)
            if goodmove:
                factor = 2*np.log(self.truealpha-1) - np.log(self.truealpha_g-1) + 2*(self.truealpha-1)*np.log(self.trueminf) - (self.truealpha_g-1)*np.log(self.trueminf_g) - \
                    self.truealpha*np.log(f1Mf) - (2*self.truealpha - self.truealpha_g)*np.log(sum_f) - np.log(gdat.sizeimag[0]*gdat.sizeimag[1]) + np.log(1. - 2./fminratio) - \
                    np.log(2*np.pi*self.kickrange_g*self.kickrange_g) + np.log(numbbrgt/(self.ng+1.-boolsplt)) + np.log((self.n-1+2*boolsplt)*weigoverpairs) + \
                    np.log(sum_f) - np.log(4.) - 2*np.log(dr2) - 3*np.log(f1Mf) - np.log(np.cos(theta)) - 3*np.log(np.sin(theta))
                if not boolsplt:
                    factor *= -1
                    factor += 2*self.penalty - self.penalty_g
                    factor += self.log_prior_moments(galaxiesb)
                else:
                    factor -= 2*self.penalty - self.penalty_g
                    factor -= self.log_prior_moments(galaxiesk)
                proposal.set_factor(factor)
            return proposal
    
        
        def stargalaxy_galaxy(self):
            boolsplt = np.random.randint(2)
            idx_reg_g = self.idx_parity_galaxies()
            sum_f = 0
            low_n = 0
            idx_bright = idx_reg_g.take(np.flatnonzero(self.galaxies[gdat.indxflux, :].take(idx_reg_g) > self.trueminf + self.trueminf_g)) # in region and bright enough to make s+g
            numbbrgt = idx_bright.size
    
            numbspmr = (self.nregx * self.nregy) / 4
            goodmove = False
            proposal = Proposal()
            # split off star
            if boolsplt and self.ng > 0 and self.n < self.nstar and numbbrgt > 0: # need something to split, but don't exceed nstar
                numbspmr = min(numbspmr, numbbrgt, self.nstar-self.n) # need bright source AND room for split off star
                dx = (np.random.normal(size=numbspmr)*self.kickrange_g).astype(np.float32)
                dy = (np.random.normal(size=numbspmr)*self.kickrange_g).astype(np.float32)
                idx_move_g = np.random.choice(idx_bright, size=numbspmr, replace=False)
                galaxies0 = self.galaxies.take(idx_move_g, axis=1)
                x0g, y0g, f0g, xx0g, xy0g, yy0g = galaxies0
                frac = (self.trueminf_g/f0g + np.random.uniform(size=numbspmr)*(1. - (self.trueminf_g + self.trueminf)/f0g)).astype(np.float32)
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
                starsb = np.empty((gdat.numbparastar, numbspmr), dtype=np.float32)
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
                numbspmr = idx_move_g.size
                goodmove = numbspmr > 0
                if goodmove:
                    proposal.add_move_galaxies(idx_move_g, galaxies0, galaxiesp)
                    proposal.add_birth_stars(starsb)
    
                # need to calculate factor
                sum_f = f0g
                invpairs = np.zeros(numbspmr)
                for k in xrange(numbspmr):
                    xtemp = self.stars[gdat.indxxpos, 0:self.n].copy()
                    ytemp = self.stars[gdat.indxypos, 0:self.n].copy()
                    xtemp = np.concatenate([xtemp, galaxiesp[gdat.indxxpos, k:k+1], starsb[gdat.indxxpos, k:k+1]])
                    ytemp = np.concatenate([ytemp, galaxiesp[gdat.indxypos, k:k+1], starsb[gdat.indxypos, k:k+1]])
    
                    invpairs[k] =  1./neighbours(xtemp, ytemp, self.kickrange_g, self.n)
            # merge star into galaxy
            elif not boolsplt and idx_reg_g.size > 1: # need two things to merge!
                numbspmr = min(numbspmr, idx_reg_g.size)
                idx_move_g = np.random.choice(idx_reg_g, size=numbspmr, replace=False) # choose galaxies and then see if they have neighbours
                idx_kill = np.empty(numbspmr, dtype=np.int)
                choosable = np.full(self.nstar, True, dtype=np.bool)
                nchoosable = float(self.nstar)
                invpairs = np.empty(numbspmr)
    
                for k in xrange(numbspmr):
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
                numbspmr = idx_move_g.size
                goodmove = numbspmr > 0
    
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
                numbspmr = idx_move_g.size
                goodmove = np.logical_and(goodmove, numbspmr > 0)
    
                if goodmove:
                    galaxiesp = np.empty((6, numbspmr))
                    galaxiesp[gdat.indxxpos,:] = frac*galaxies0[gdat.indxxpos,:] + (1-frac)*starsk[gdat.indxxpos,:]
                    galaxiesp[gdat.indxypos,:] = frac*galaxies0[gdat.indxypos,:] + (1-frac)*starsk[gdat.indxypos,:]
                    galaxiesp[gdat.indxflux,:] = galaxies0[gdat.indxflux,:] + starsk[gdat.indxflux,:]
                    galaxiesp[self._XX,:] = pxxg
                    galaxiesp[self._XY,:] = pxyg
                    galaxiesp[self._YY,:] = pyyg
                    proposal.add_move_galaxies(idx_move_g, galaxies0, galaxiesp)
                    proposal.add_death_stars(idx_kill, starsk)
                # this proposal makes a galaxy that is bright enough to split
                numbbrgt = numbbrgt + 1
            if goodmove:
                factor = np.log(self.truealpha-1) - (self.truealpha-1)*np.log(sum_f/self.trueminf) - self.truealpha_g*np.log(frac) - self.truealpha*np.log(1-frac) + \
                        np.log(2*np.pi*self.kickrange_g*self.kickrange_g) - np.log(gdat.sizeimag[0]*gdat.sizeimag[1]) + np.log(1. - (self.trueminf+self.trueminf_g)/sum_f) + \
                        np.log(numbbrgt/float(self.ng)) + np.log((self.n+1.-boolsplt)*invpairs) - 2*np.log(frac)
                if not boolsplt:
                    factor *= -1
                factor += self.log_prior_moments(galaxiesp) - self.log_prior_moments(galaxies0) # galaxy prior
                if not boolsplt:
                    factor += self.penalty
                else:
                    factor -= self.penalty
                proposal.set_factor(factor)
            return proposal
    
        
        def merge_split_galaxies(self):
            boolsplt = np.random.randint(2)
            idx_reg = self.idx_parity_galaxies()
            sum_f = 0
            low_n = 0
            idx_bright = idx_reg.take(np.flatnonzero(self.galaxies[gdat.indxflux, :].take(idx_reg) > 2*self.trueminf_g)) # in region!
            numbbrgt = idx_bright.size
    
            numbspmr = (self.nregx * self.nregy) / 4
            goodmove = False
            proposal = Proposal()
            # split
            if boolsplt and self.ng > 0 and self.ng < self.ngalx and numbbrgt > 0: # need something to split, but don't exceed nstar
                numbspmr = min(numbspmr, numbbrgt, self.ngalx-self.ng) # need bright source AND room for split source
                dx = (np.random.normal(size=numbspmr)*self.kickrange_g).astype(np.float32)
                dy = (np.random.normal(size=numbspmr)*self.kickrange_g).astype(np.float32)
                idx_move_g = np.random.choice(idx_bright, size=numbspmr, replace=False)
                galaxies0 = self.galaxies.take(idx_move_g, axis=1)
                fminratio = galaxies0[gdat.indxflux,:] / self.trueminf_g
                frac = (1./fminratio + np.random.uniform(size=numbspmr)*(1. - 2./fminratio)).astype(np.float32)
                frac_xx = np.random.uniform(size=numbspmr).astype(np.float32)
                frac_xy = np.random.uniform(size=numbspmr).astype(np.float32)
                frac_yy = np.random.uniform(size=numbspmr).astype(np.float32)
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
                numbspmr = idx_move_g.size
                goodmove = numbspmr > 0
    
                # need to calculate factor
                sum_f = galaxies0[gdat.indxflux,:]
                invpairs = np.empty(numbspmr)
                for k in xrange(numbspmr):
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
            elif not boolsplt and idx_reg.size > 1: # need two things to merge!
                numbspmr = min(numbspmr, idx_reg.size/2)
                idx_move_g = np.empty(numbspmr, dtype=np.int)
                idx_kill_g = np.empty(numbspmr, dtype=np.int)
                choosable = np.zeros(self.ngalx, dtype=np.bool)
                choosable[idx_reg] = True
                nchoosable = float(idx_reg.size)
                invpairs = np.empty(numbspmr)
    
                for k in xrange(numbspmr):
                    idx_move_g[k] = np.random.choice(self.ngalx, p=choosable/nchoosable)
                    invpairs[k], idx_kill_g[k] = neighbours(self.galaxies[gdat.indxxpos, 0:self.ng], \
                                                            self.galaxies[gdat.indxypos, 0:self.ng], self.kickrange_g, idx_move_g[k], generate=True)
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
    
                numbspmr = idx_move_g.size
                goodmove = numbspmr > 0
                if goodmove:
                    proposal.add_move_galaxies(idx_move_g, galaxies0, galaxiesp)
                    proposal.add_death_galaxies(idx_kill_g, galaxiesk)
    
                # turn numbbrgt into an array
                numbbrgt = numbbrgt - (galaxies0[gdat.indxflux,:] > 2*self.trueminf_g) - (galaxiesk[gdat.indxflux,:] > 2*self.trueminf_g) + \
                                                                                                                (galaxiesp[gdat.indxflux,:] > 2*self.trueminf_g)
            if goodmove:
                factor = np.log(self.truealpha_g-1) + (self.truealpha_g-1)*np.log(self.trueminf) - self.truealpha_g*np.log(frac*(1-frac)*sum_f) + \
                    np.log(2*np.pi*self.kickrange_g*self.kickrange_g) - np.log(gdat.sizeimag[0]*gdat.sizeimag[1]) + np.log(1. - 2./fminratio) + np.log(numbbrgt) + np.log(invpairs) + \
                    np.log(sum_f) + np.log(xx_p) + np.log(np.abs(xy_p)) + np.log(yy_p) - 3*np.log(frac) - 3*np.log(1-frac) # last line is Jacobian
                if not boolsplt:
                    factor *= -1
                    factor += self.penalty_g
                    factor += self.log_prior_moments(galaxiesp) - self.log_prior_moments(galaxies0) - self.log_prior_moments(galaxiesk)
                else:
                    factor -= self.penalty_g
                    factor += self.log_prior_moments(galaxiesp) - self.log_prior_moments(galaxies0) + self.log_prior_moments(galaxiesb)
                proposal.set_factor(factor)
            return proposal

    ntemps = 1
    temps = np.sqrt(2) ** np.arange(ntemps)
    
    print 'gdat.sizeimag'
    print gdat.sizeimag
    print ''
    assert gdat.sizeimag[0] % sizeregi == 0
    assert gdat.sizeimag[1] % sizeregi == 0
    margin = 10
    
    nstar = Model.nstar
    numbstarsamp = np.zeros(gdat.numbsamp, dtype=np.int32)
    xpossamp = np.zeros((gdat.numbsamp, nstar), dtype=np.float32)
    ypossamp = np.zeros((gdat.numbsamp, nstar), dtype=np.float32)
    fluxsamp = np.zeros((gdat.numbsamp, nstar), dtype=np.float32)
    
    if gdat.booltimebins:
        lcprsamp = np.zeros((gdat.numbsamp, gdat.numblcpr, nstar), dtype=np.float32)
    ngsample = np.zeros(gdat.numbsamp, dtype=np.int32)
    xgsample = np.zeros((gdat.numbsamp, nstar), dtype=np.float32)
    ygsample = np.zeros((gdat.numbsamp, nstar), dtype=np.float32)
    fgsample = np.zeros((gdat.numbsamp, nstar), dtype=np.float32)
    xxgsample = np.zeros((gdat.numbsamp, nstar), dtype=np.float32)
    xygsample = np.zeros((gdat.numbsamp, nstar), dtype=np.float32)
    yygsample = np.zeros((gdat.numbsamp, nstar), dtype=np.float32)
    
    # construct model for each temperature
    models = [Model() for k in xrange(ntemps)]
    
    # write the chain
    ## h5 file path
    pathh5py = gdat.pathdatartag + gdat.rtag + '_chan.h5'
    ## numpy object file path
    pathnump = gdat.pathdatartag + gdat.rtag + '_chan.npz'
    
    filearry = h5py.File(pathh5py, 'w')
    print 'Will write the chain to %s...' % pathh5py
    
    if boolplot:
        plt.figure(figsize=(21, 7))
    
    for j in xrange(gdat.numbsamp):
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
    
        numbstarsamp[j] = models[0].n
        xpossamp[j,:] = models[0].stars[gdat.indxxpos, :]
        ypossamp[j,:] = models[0].stars[gdat.indxypos, :]
        fluxsamp[j,:] = models[0].stars[gdat.indxflux, :]
        if gdat.booltimebins:
            lcprsamp[j, :, :] = models[0].stars[gdat.indxlcpr, :]
            if gdat.diagmode:
                chec_lcpr(models[0].stars[gdat.indxlcpr, :])
        
        if strgmodl == 'galx':
            ngsample[j] = models[0].ng
            xgsample[j,:] = models[0].galaxies[gdat.indxxpos, :]
            ygsample[j,:] = models[0].galaxies[gdat.indxypos, :]
            fgsample[j,:] = models[0].galaxies[gdat.indxflux, :]
            xxgsample[j,:] = models[0].galaxies[Model._XX, :]
            xygsample[j,:] = models[0].galaxies[Model._XY, :]
            yygsample[j,:] = models[0].galaxies[Model._YY, :]
   
    # burn out the initial numbsampburn samples
    numbstarsamp = numbstarsamp[gdat.numbsampburn:]
    xpossamp = xpossamp[gdat.numbsampburn:, :]
    ypossamp = ypossamp[gdat.numbsampburn:, :]
    fluxsamp = fluxsamp[gdat.numbsampburn:, :]
    if gdat.booltimebins:
        lcprsamp = lcprsamp[gdat.numbsampburn:, :, :]
    
    filearry.create_dataset('numbstar', data=numbstarsamp)
    filearry.create_dataset('xpos', data=xpossamp)
    filearry.create_dataset('ypos', data=ypossamp)
    filearry.create_dataset('flux', data=fluxsamp)
    if gdat.booltimebins:
        filearry.create_dataset('lcpr', data=lcprsamp)
        if gdat.diagmode:
            chec_lcpr(lcprsamp)
    filearry.close()

    path = gdat.pathdatartag + 'gdat.p'
    filepick = open(path, 'wb')
    print 'Writing to %s...' % path
    cPickle.dump(gdat, filepick, protocol=cPickle.HIGHEST_PROTOCOL)
    filepick.close()
 
    print 'Saving the numpy object to %s...' % pathnump
    np.savez(pathnump, n=numbstarsamp, x=xpossamp, y=ypossamp, f=fluxsamp, ng=ngsample, xg=xgsample, yg=ygsample, fg=fgsample, xxg=xxgsample, xyg=xygsample, yyg=yygsample)
    
    # calculate the condensed catalog
    catlcond = retr_catlcond(gdat.rtag, gdat.pathdata)
    
    if gdat.booltimebins:
        gdat.numbsourcond = catlcond.shape[0]
        gdat.indxsourcond = np.arange(gdat.numbsourcond)
        plot_lcur(gdat)

    if boolplot:
        
        # plot the condensed catalog
        if boolplotsave:
            figr, axis = plt.subplots()
            if gdat.booltimebins:
                temp = gdat.cntpdata[0, :, :]
            else:
                temp = gdat.cntpdata
            axis.imshow(temp, origin='lower', interpolation='none', cmap='Greys', vmin=np.min(gdat.cntpdata), vmax=np.percentile(gdat.cntpdata, 95))
            numbsampplot = min(gdat.numbsamp - gdat.numbsampburn, 10)
            print 'gdat.numbsamp'
            print gdat.numbsamp
            print 'gdat.numbsampburn'
            print gdat.numbsampburn
            indxsampplot = np.random.choice(np.arange(gdat.numbsampburn, gdat.numbsamp, dtype=int), size=numbsampplot, replace=False) 
            for k in range(numbsampplot):
                print 'k'
                print k
                print 'indxsampplot'
                summgene(indxsampplot)
                print 'numbstarsamp'
                summgene(numbstarsamp)
                print
                numb = numbstarsamp[indxsampplot[k]]
                supr_catl(gdat, axis, xpossamp[k, :numb], ypossamp[k, :numb], fluxsamp[k, :numb])
            supr_catl(gdat, axis, catlcond[:, 0], catlcond[:, 2], catlcond[:, 4], boolcond=True)
            plt.savefig(gdat.pathdatartag + '%s_condcatl.' % gdat.rtag + gdat.strgplotfile)

    if boolplotsave:
        print 'Making the animation...'
        cmnd = 'convert -delay 20 -density 200x200 %s/%s_cntpdata*.%s %s/%s_cntpdata.gif' % (gdat.pathdatartag, gdat.rtag, gdat.strgplotfile, gdat.pathdatartag, gdat.rtag)
        print cmnd
        os.system(cmnd)
        cmnd = 'convert -delay 20 -density 200x200 %s/%s_cntpresi*.%s %s/%s_cntpresi.gif' % (gdat.pathdatartag, gdat.rtag, gdat.strgplotfile, gdat.pathdatartag, gdat.rtag)
        print cmnd
        os.system(cmnd)
    

def chec_lcpr(lcpr):

    if np.amax(np.abs(lcpr)) > 1e3 or np.isnan(lcpr).any():
        if np.amax(np.abs(lcpr)) > 1e3:
            print 'if np.amax(np.abs(lcpr)) > 1e3'
        if np.isnan(lcpr).any():
            print 'np.isnan(lcpr).any()'
            for k in range(lcpr.shape[1]):
                print lcpr[:, k]
        print 'lcpr'
        summgene(lcpr)
        print lcpr
        print 

        raise Exception('')


def retr_catlseed(rtag, pathdata):
    
    strgtimestmp = rtag[:15]
    
    pathlion, pathdata = retr_path(pathdata)
    
    pathdatartag = pathdata + rtag + '/'
    path = pathdatartag + 'gdat.p'
    filepick = open(path, 'rb')
    print 'Reading %s...' % path
    gdat = cPickle.load(filepick)
    filepick.close()

    os.system('mkdir -p %s' % gdat.pathdatartag)

    # maximum number of sources
    maxmnumbsour = 2000
    
    # number of samples used in the seed catalog determination
    numbsampseed = 10
    pathchan = gdat.pathdatartag + rtag + '_chan.h5'
    filechan = h5py.File(pathchan, 'r')
    xpossamp = filechan['xpos'][()][:numbsampseed, :] 
    ypossamp = filechan['ypos'][()][:numbsampseed, :]
    fluxsamp = filechan['flux'][()][:numbsampseed, :]
    if gdat.booltimebins:
        lcprsamp = filechan['lcpr'][()][:numbsampseed, :, :]
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

    np.savetxt(gdat.pathdatartag + rtag + '_catlseed.txt', catlseed)


def retr_catlcond(rtag, pathdata):

    strgtimestmp = rtag[:15]
    
    # paths
    pathlion, pathdata = retr_path(pathdata)
    
    pathdatartag = pathdata + rtag + '/'
    os.system('mkdir -p %s' % pathdatartag)

    pathcatlcond = pathdatartag + rtag + '_catlcond.h5'
    
    # search radius
    radisrch = 0.75
    
    # confidence cut
    cut = 0. 
    
    # gain
    gain = 0.00546689
    
    path = pathdatartag + 'gdat.p'
    filepick = open(path, 'rb')
    print 'Reading %s...' % path
    gdat = cPickle.load(filepick)
    filepick.close()
    
    # read the chain
    print 'Reading the chain...'    
    pathchan = pathdatartag + rtag + '_chan.h5'
    filechan = h5py.File(pathchan, 'r')
    catlxpos = filechan['xpos'][()] 
    catlypos = filechan['ypos'][()]
    catlflux = filechan['flux'][()]
    if gdat.booltimebins:
        catllcpr = filechan['lcpr'][()]
        numblcpr = catllcpr.shape[0]

    numbsamp = len(catlxpos)
    catlnumb = np.zeros(numbsamp, dtype=int)
    gdat.indxsamp = np.arange(numbsamp)
    for k in gdat.indxsamp:
        catlnumb[k] = len(catlxpos[k])
    filechan.close()
    
    maxmnumbsour = catlxpos.shape[1]
    
    # sort the catalog in decreasing flux
    catlsort = np.zeros((numbsamp, maxmnumbsour, gdat.numbparastar))
    for i in gdat.indxsamp:
        catl = np.zeros((maxmnumbsour, gdat.numbparastar))
        catl[:, gdat.indxxpos] = catlxpos[i, :]
        catl[:, gdat.indxypos] = catlypos[i, :]
        catl[:, gdat.indxflux] = catlflux[i, :] 
        if gdat.booltimebins:
            catl[:, gdat.indxlcpr] = catllcpr[i, :, :].T 
        catlsort[i, :, :] = np.flipud(catl[catl[:, gdat.indxflux].argsort()])
    
    print "Stacking catalogs..."
    
    # create array for KD tree creation
    PCc_stack = np.zeros((np.sum(catlnumb), 2))
    j = 0
    for i in xrange(catlnumb.size):
        n = catlnumb[i]
        PCc_stack[j:j+n, 0] = catlsort[i, 0:n, gdat.indxxpos]
        PCc_stack[j:j+n, 1] = catlsort[i, 0:n, gdat.indxypos]
        j += n

    retr_catlseed(rtag, gdat.pathdata)
    
    # seed catalog
    ## load the catalog
    pathcatlseed = gdat.pathdatartag + rtag + '_catlseed.txt'
    data = np.loadtxt(pathcatlseed)
    
    ## perform confidence cut
    seedxpos = data[:,0][data[:,2] >= cut*300]
    seedypos = data[:,1][data[:,2] >= cut*300]
    seednumb = data[:,2][data[:,2] >= cut*300]
    
    assert seedxpos.size == seedypos.size
    assert seedxpos.size == seednumb.size
    
    catlseed = np.zeros((seedxpos.size, 2))
    catlseed[:, 0] = seedxpos
    catlseed[:, 1] = seedypos
    numbsourseed = seedxpos.size
    numbsourcatlseed = len(catlseed)
    indxsourcatlseed = np.arange(numbsourcatlseed)
    
    #creates tree, where tree is Pcc_stack
    tree = scipy.spatial.KDTree(PCc_stack)
    
    # features of the condensed sources
    featcond = np.zeros((numbsamp, numbsourcatlseed, gdat.numbparastar))
    
    #numpy mask for sources that have been matched
    mask = np.zeros(len(PCc_stack))
    
    #first, we iterate over all sources in the seed catalog:
    for ct in indxsourcatlseed:
    
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
    
                    #add information to cluster array
                    featcond[i, ct, gdat.indxxpos] = catlsort[i, match-cat_lo_ndx, gdat.indxxpos]
                    featcond[i, ct, gdat.indxypos] = catlsort[i, match-cat_lo_ndx, gdat.indxypos]
                    featcond[i, ct, gdat.indxflux] = catlsort[i, match-cat_lo_ndx, gdat.indxflux]
                    if gdat.booltimebins:
                        featcond[i, ct, gdat.indxlcpr] = catlsort[i, match-cat_lo_ndx, gdat.indxlcpr]
    
    # generate condensed catalog from clusters
    numbsourseed = len(catlseed)
    
    #arrays to store 'classical' catalog parameters
    xposmean = np.zeros(numbsourseed)
    yposmean = np.zeros(numbsourseed)
    fluxmean = np.zeros(numbsourseed)
    magtmean = np.zeros(numbsourseed)
    if gdat.booltimebins:
        lcprmean = np.zeros((gdat.numblcpr, numbsourseed))
    stdvxpos = np.zeros(numbsourseed)
    stdvypos = np.zeros(numbsourseed)
    stdvflux = np.zeros(numbsourseed)
    if gdat.booltimebins:
        stdvlcpr = np.zeros((gdat.numblcpr, numbsourseed))
    stdvmagt = np.zeros(numbsourseed)
    conf = np.zeros(numbsourseed)
    
    # confidence interval defined for err_(x,y,f)
    pctlhigh = 84
    pctlloww = 16
    for i in indxsourcatlseed:
        xpos = featcond[:, i, gdat.indxxpos][np.nonzero(featcond[:, i, gdat.indxxpos])]
        ypos = featcond[:, i, gdat.indxypos][np.nonzero(featcond[:, i, gdat.indxypos])]
        flux = featcond[:, i, gdat.indxflux][np.nonzero(featcond[:, i, gdat.indxflux])]
        if gdat.booltimebins:
            lcpr = featcond[:, i, gdat.indxlcpr][np.nonzero(featcond[:, i, gdat.indxlcpr])]
        
        assert xpos.size == ypos.size
        assert xpos.size == flux.size
        
        conf[i] = xpos.size/300.0
            
        if gdat.booltimebins and lcpr.size > 0:
            xposmean[i] = np.mean(xpos)
            yposmean[i] = np.mean(ypos)
            fluxmean[i] = np.mean(flux)
            lcprmean[:, i] = np.mean(lcpr, axis=0)
        if xpos.size > 1:
            stdvxpos[i] = np.percentile(xpos, pctlhigh) - np.percentile(xpos, pctlloww)
            stdvypos[i] = np.percentile(ypos, pctlhigh) - np.percentile(ypos, pctlloww)
            stdvflux[i] = np.percentile(flux, pctlhigh) - np.percentile(flux, pctlloww)
            if gdat.booltimebins:
                stdvlcpr[:, i] = np.percentile(lcpr, pctlhigh) - np.percentile(lcpr, pctlloww)

    catlcond = np.zeros((gdat.numbparacatlcond, numbsourseed, 2))
    catlcond[gdat.indxxpos, :, 0] = xposmean
    catlcond[gdat.indxxpos, :, 1] = stdvxpos
    catlcond[gdat.indxypos, :, 0] = yposmean
    catlcond[gdat.indxypos, :, 1] = stdvypos
    catlcond[gdat.indxflux, :, 0] = fluxmean
    catlcond[gdat.indxflux, :, 1] = stdvflux
    if gdat.booltimebins:
        catlcond[gdat.indxlcpr, :, 0] = lcprmean
        catlcond[gdat.indxlcpr, :, 1] = stdvlcpr
    catlcond[gdat.indxconf, :, 0] = conf
    
    #magt = 22.5 - 2.5 * np.log10(flux * gain)

    ## h5 file path
    print 'Will write the chain to %s...' % pathcatlcond
    filecatlcond = h5py.File(pathcatlcond, 'w')
    filecatlcond.create_dataset('catlcond', data=catlcond)
    filecatlcond.close()

    return catlcond


def retr_path(pathdata=None):
    
    pathlion = os.environ['LION_PATH'] + '/'
    if pathdata == None:
        pathdata = os.environ['LION_DATA_PATH'] + '/'
    
    return pathlion, pathdata


# configurations

def read_datafromtext(strgdata):
    
    pathlion, pathdata = retr_path()

    filepixl = open(pathdata + strgdata + '_pixl.txt')
    filepixl.readline()
    bias, gain = [np.float32(i) for i in filepixl.readline().split()]
    filepixl.close()
    cntpdata = np.loadtxt(pathdata + strgdata + '_cntp.txt').astype(np.float32)
    cntpdata -= bias
    
    return cntpdata, bias, gain


def cnfg_defa():
    
    # read the data
    strgdata = 'sdss0921'
    cntpdata, bias, gain = read_datafromtext(strgdata)
    
    # read PSF
    pathlion, pathdata = retr_path()
    filepsfn = open(pathdata + strgdata + '_psfn.txt')
    numbsidepsfn, factusam = [np.int32(i) for i in filepsfn.readline().split()]
    filepsfn.close()
    cntppsfn = np.loadtxt(pathdata + strgdata + '_psfn.txt', skiprows=1).astype(np.float32)
    
    print 'cntppsfn'
    summgene(cntppsfn)
    print 'cntpdata'
    summgene(cntpdata)
    print

    strgmodl = 'star'
    datatype = 'mock'
    catlrefr = [{}]

    if datatype == 'mock':
        
        # get the true catalog 
        if strgmodl == 'star':
            truth = np.loadtxt(pathdata + strgdata + '_true.txt')
            catlrefr[0]['xpos'] = truth[:, 0]
            catlrefr[0]['ypos'] = truth[:, 1]
            catlrefr[0]['flux'] = truth[:, 2]
        if strgmodl == 'galx':
            truth_s = np.loadtxt(pathdata + strgdata + '_str.txt')
            catlrefr['xpos'] = truth_s[:, 0]
            catlrefr['ypos'] = truth_s[:, 1]
            catlrefr['flux'] = truth_s[:, 2]
            truth_g = np.loadtxt(pathdata + strgdata + '_gal.txt')
            truexg = truth_g[:,0]
            trueyg = truth_g[:,1]
            truefg = truth_g[:,2]
            truexxg= truth_g[:,3]
            truexyg= truth_g[:,4]
            trueyyg= truth_g[:,5]
            truerng, theta, phi = from_moments(truexxg, truexyg, trueyyg)
        if strgmodl == 'stargalx':
            truth = np.loadtxt(pathdata + 'truecnts.txt')
            filetrue = h5py.File(pathdata + 'true.h5', 'r')
            for attr in filetrue:
                gdat[attr] = filetrue[attr][()]
            filetrue.close()
        
        lablrefr = ['True']
        colrrefr = ['g']
    else:
        lablrefr = ['HST 606W']
        colrrefr = ['m']
    

    main( \
         cntpdata=cntpdata, \
         cntppsfn=cntppsfn, \
         factusam=factusam, \

         catlrefr=catlrefr, \
         lablrefr=lablrefr, \
         colrrefr=colrrefr, \

         #numbsamp=200, \
         colrstyl='pcat', \
         #boolplotsave=False, \
         #diagmode=True, \
         boolplotshow=False, \
         boolplotsave=True, \
         testpsfn=True, \
         bias=bias, \
         gain=gain, \
         verbtype=1, \
         #numbsamp=5, \
         #numbloop=10, \
         #booltimebins=True, \
        )


def cnfg_time():
    
    # read the data
    strgdata = 'sdss0921'
    cntpdata, bias, gain = read_datafromtext(strgdata)
   
    # perturb the data in each time bin
    numbtime = 10
    indxtime = np.arange(numbtime)
    numbsidexpos = cntpdata.shape[1]
    numbsideypos = cntpdata.shape[0]
    numbpixl = numbsidexpos * numbsideypos
    cntpdata = np.tile(cntpdata, (numbtime, 1, 1))
    for k in indxtime:
        cntpdata[k, :, :] += (4 * np.random.randn(numbpixl).reshape((numbsidexpos, numbsideypos))).astype(int)

    # read PSF
    pathlion, pathdata = retr_path()
    filepsfn = open(pathdata + strgdata + '_psfn.txt')
    numbsidepsfn, factusam = [np.int32(i) for i in filepsfn.readline().split()]
    filepsfn.close()
    cntppsfn = np.loadtxt(pathdata + strgdata + '_psfn.txt', skiprows=1).astype(np.float32)

    main( \
         cntpdata=cntpdata, \
         cntppsfn=cntppsfn, \
         factusam=factusam, \
         numbsamp=100, \
         numbloop=1000, \
         colrstyl='pcat', \
         #boolplotsave=False, \
         boolplotsave=True, \
         boolplotshow=False, \
         booltimebins=True, \
         diagmode=True, \
         bias=bias, \
         gain=gain, \
         #verbtype=2, \
         #testpsfn=True, \
         #boolplotsave=True, \
         #boolplotshow=True, \
        )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    else:
        globals().get(sys.argv[1])()





