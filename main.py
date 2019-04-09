#import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double
import h5py, datetime
import matplotlib 

from astropy.wcs import WCS

# temp
import seaborn as sns
sns.set(context='poster', style='ticks', color_codes=True)

from scipy import ndimage

import copy

import matplotlib.pyplot as plt

import cPickle, inspect

import scipy.spatial
import networkx

import time
import astropy.wcs
import astropy.io.fits as fits

import scipy.misc

import sys, os, warnings

from __init__ import *

class gdatstrt(object):

    def __init__(self):
        pass
    
    def __setattr__(self, attr, valu):
        super(gdatstrt, self).__setattr__(attr, valu)


def plot_varbsamp(gdat, varb, lablyaxi, strgplot, xaxitype='samp'):

    figr, axis = plt.subplots()
    
    if xaxitype == 'samp':
        indx = gdat.indxsamp
    elif xaxitype == 'swlp':
        indx = gdat.indxswlp
    elif xaxitype == 'swep':
        indx = gdat.indxswep
    
    axis.plot(indx, varb)
    if xaxitype == 'samp':
        axis.set_xlabel('$i_{samp}$')
    elif xaxitype == 'swep':
        axis.set_xlabel('$i_{swep}$')
    elif xaxitype == 'swlp':
        axis.set_xlabel('$i_{swlp}$')
    axis.set_ylabel(lablyaxi)
    plt.tight_layout()
    plt.savefig(gdat.pathdatartag + '%s.%s' % (strgplot, gdat.strgplotfile))
    plt.close()


def supr_catl(gdat, axis, i, t, xpos=None, ypos=None, flux=None, boolcond=False, plottext=False, colrmodl='b', shapmodl='x'):
    markers = ['+', '1'] # temp
    for k in gdat.indxrefr:
        # minimum size of gdat.sizemrkr at 100 ADU
        #size = gdat.sizemrkr * (np.log10(gdat.catlrefr[k]['flux'][i, t, :]) - 1.)
        #size[np.where(size < gdat.sizemrkr)] = gdat.sizemrkr
        size = gdat.sizemrkr * gdat.catlrefr[k]['flux'][i, t, :] / gdat.fittminmflux
        axis.scatter(gdat.catlrefr[k]['xpos'], gdat.catlrefr[k]['ypos'], marker=markers[k], s=size, color=gdat.colrrefr[k], lw=1, alpha=0.7)
        if plottext:
            for n in range(len(gdat.catlrefr[k]['xpos'])):
                if gdat.catlrefr[k]['strg'][n] != None:
                    axis.text(gdat.catlrefr[k]['xpos'][n] - 3., gdat.catlrefr[k]['ypos'][n] - 3., gdat.catlrefr[k]['strg'][n], \
                                                        horizontalalignment='center', verticalalignment='center', color=gdat.colrrefr[k])
            
            #if boolcond:
            #    for n in range(len(gdat.catlrefr[k]['xpos'])):
            #        axis.text(gdat.catlrefr[k]['xpos'][n] - 3., gdat.catlrefr[k]['ypos'][n] - 3., str(n), \
            #                        #horizontalalignment='center', verticalalignment='center', transform=axis.transAxes, color='g')
            #                        horizontalalignment='center', verticalalignment='center', color='g')
    
    if xpos is not None:
        if boolcond:
            colr = 'yellow'
        else:
            colr = colrmodl
        #size = gdat.sizemrkr * (np.log10(flux) - 1.)
        #size[np.where(size < gdat.sizemrkr)] = gdat.sizemrkr
        size = gdat.sizemrkr * flux / gdat.fittminmflux
        axis.scatter(xpos, ypos, marker=shapmodl, s=size, color=colr, lw=1, alpha=0.7)
        if boolcond:
            for n in range(len(xpos)):
                axis.text(xpos[n] + 1., ypos[n] + 1., str(n), \
                                horizontalalignment='center', verticalalignment='center', color=colr, fontsize=6)
    
        
def plot_pcat(gdat=None, rtag=None):
    
    pathlion, pathdata = retr_path()
    
    if gdat is None:
        path = pathdata + rtag + '/gdat.p'
        filepick = open(path, 'rb')
        print 'Reading %s...' % path
        gdat = cPickle.load(filepick)
        filepick.close()
    
    setp(gdat)
    
    psf0 = gdat.cntppsfnusam[:gdat.numbsidepsfnusam:gdat.factusam, :gdat.numbsidepsfnusam:gdat.factusam]
    
    print 'gdat.strgdata'
    print gdat.strgdata
    # if mock data, find mock catalogue, otherwise assume first reference catalogue is ground truth
    if gdat.datatype == 'mock':
        path = pathdata + '%04d%04d%04d_mock.h5' % (gdat.numbener, gdat.numbtime, gdat.truenumbstar)
        print 'Reading %s...' % path
        filetemp = h5py.File(path, 'r')
        HTcat = read_catl(gdat, path, boolmock=True)
    else:
        HTcat = gdat.catlrefr[0]
    
    HTx = HTcat['xpos']
    HTy = HTcat['ypos']
    HTmag = HTcat['magt'][0, 0, :] # temp
    HTc = np.zeros((HTx.shape[0], 2))
    HTc[:, 0] = HTx
    HTc[:, 1] = HTy
    HTkd = scipy.spatial.KDTree(HTc)

    kdtrrefr = []
    magtrefr = []
    for a in xrange(1, len(gdat.catlrefr)):
        catlthis = gdat.catlrefr[a]
        catlcord = np.dstack((catlthis['xpos'], catlthis['ypos']))[0]
        kdtrrefr.append(scipy.spatial.KDTree(catlcord))
        magtrefr.append(catlthis['magt'][0,0,:])
    
    path = gdat.pathdatartag + gdat.rtag + '_catlcond.h5'
    filetemp = h5py.File(path, 'r')
    print 'Reading %s...' % path
    catlcond = filetemp['catlcond'][()]
    filetemp.close()
    CCx = catlcond[0, :, 0]
    CCy = catlcond[1, :, 0]
    CCf = catlcond[2, :, 0]
    CCmag = gdat.magtzero - 2.5*np.log10(CCf)
    CCconf = catlcond[gdat.indxprvl, :, 0]
    print 'CCconf'
    summgene(CCconf)
    CCs = catlcond[0, :, 0]
    
    CCc = np.zeros((CCx.shape[0], 2))
    if gdat.numbsourcond > 0:
        CCc[:,0] = CCx
        CCc[:,1] = CCy
        CCkd = scipy.spatial.KDTree(CCc)
    
    pathchan = gdat.pathdatartag + gdat.rtag + '_chan.h5'
    chan = read_catl(gdat, pathchan)
    maxn = 3000
    PCn = chan['numb']
    PCx = chan['xpos']
    PCy = chan['ypos']
    PCf = chan['flux'][:, 0, 0, :]
    PCmag = gdat.magtzero - 2.5*np.log10(PCf)
    mask = (PCx > 0) * (PCy > 0) * (PCf > 0)
    PCc_all = np.zeros((np.sum(mask), 2))
    PCc_all[:, 0] = PCx[mask].flatten()
    PCc_all[:, 1] = PCy[mask].flatten()
    PCkd = scipy.spatial.KDTree(PCc_all)
    PCmag_all = PCmag[mask]
    
    sizefac = 1360.
    n = PCn[-1]
    
    dr = 0.75
    dmag = 0.5
    
    nbins = 16
    minr, maxr = np.amin(HTmag) - 0.1, gdat.magtzero - 2.5*np.log10(gdat.fittminmflux)
    binw = (maxr - minr) / float(nbins)
    #print "max r CC", np.max(CCf)
    
    precPC = np.zeros(nbins)
    precCC = np.zeros(nbins)
    if len(gdat.catlrefr) > 1:
        precrefr = np.zeros((len(gdat.catlrefr) - 1, nbins))
        goodmtchrefr = []

    # with sigfac < 8
    
    gdat.distmtch, gdat.indxmtch = CCkd.query(np.vstack((HTx, HTy)).T, k=1, distance_upper_bound=dr, p=2)
    if gdat.verbtype > 1:
        print 'HTx'
        print HTx
        print 'HTy'
        print HTy
        print 'HTf'
        print HTf
        print 'CCx'
        print CCx
        print 'CCy'
        print CCy
        print 'CCf'
        print CCf
        print 'gdat.indxmtch'
        print gdat.indxmtch
        print

    goodmatchCC, indxmtch = associate(CCkd, CCmag, HTkd, HTmag, dr, dmag)
    goodmatchPC, indxmtch = associate(PCkd, PCmag_all, HTkd, HTmag, dr, dmag)
    for a in xrange(len(gdat.catlrefr) - 1):
        goodmtchthis, indxmtch = associate(kdtrrefr[a], magtrefr[a], HTkd, HTmag, dr, dmag)
        goodmtchrefr.append(goodmtchthis)
    
    for i in xrange(nbins):
        rlo = minr + i * binw
        rhi = rlo + binw
    
        inbin = np.logical_and(PCmag_all >= rlo, PCmag_all < rhi)
        precPC[i] = np.sum(np.logical_and(inbin, goodmatchPC)) / float(np.sum(inbin))
    
        inbin = np.logical_and(CCmag >= rlo, CCmag < rhi)
        precCC[i] = np.sum(CCconf[np.logical_and(inbin, goodmatchCC)]) / float(np.sum(CCconf[inbin]))

        # loop only runs if more than one reference catalogue; assumes catalogues after first are not ground truth
        for a in xrange(len(gdat.catlrefr) - 1):
            inbin = np.logical_and(magtrefr[a] >= rlo, magtrefr[a] < rhi)
            precrefr[a,i] = np.sum(np.logical_and(inbin, goodmtchrefr[a])) / float(np.sum(inbin))

    plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precPC, c='r', label='Catalog Ensemble', marker='x', markersize=10, mew=2)
    plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precCC, c='purple', label='Condensed Catalog', marker='1', markersize=10, mew=2)
    for a in xrange(len(gdat.catlrefr) - 1):
        plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precrefr[a,:], c=gdat.colrrefr[a+1], label=gdat.lablrefr[a+1], marker='+', markersize=10, mew=2)
    plt.xlabel(gdat.lablrefr[0] + ' magnitude')
    plt.ylabel('false discovery rate')
    plt.legend(prop={'size':12}, loc = 'best')
    plt.tight_layout()
    plt.savefig(gdat.pathdatartag + 'fldr.pdf')
    plt.close()
    
    completeCC, indxmtchtrue, confmatchCC, sigfCC = associate(HTkd, HTmag, CCkd, CCmag, dr, dmag, confs_b=CCconf, sigfs_b=CCs)
    
    completePC = np.zeros((PCx.shape[0], HTx.size))
    for i in xrange(PCx.shape[0]):
        n = PCn[i]
        CCc_one = np.zeros((n,2))
        CCc_one[:, 0] = PCx[i, 0:n]
        CCc_one[:, 1] = PCy[i, 0:n]
        CCf_one = PCmag[i, 0:n]
        plt.scatter(PCx[i,0:n], PCy[i,0:n], marker='x', c='r', s=10**(-0.4*(PCmag[i,0:n]-21)))
        plt.scatter(HTx, HTy, marker='+', c='g', s=10**(-0.4*(HTmag-21)))
        plt.close()

        outp, indxmtchtrue = associate(HTkd, HTmag, scipy.spatial.KDTree(CCc_one), CCf_one, dr, dmag)
        completePC[i, :] = outp
    completePC = np.sum(completePC, axis=0) / float(PCx.shape[0])
    
    reclPC = np.zeros(nbins)
    reclCC = np.zeros(nbins)
    if len(gdat.catlrefr) > 1:
        reclrefr = np.zeros((len(gdat.catlrefr) - 1, nbins))
        cmplrefr = []
    for a in xrange(len(gdat.catlrefr) - 1):
        cmplthis, indxmtchtrue = associate(HTkd, HTmag, kdtrrefr[a], magtrefr[a], dr, dmag)
        cmplrefr.append(cmplthis)
    
    for i in xrange(nbins):
        rlo = minr + i * binw
        rhi = rlo + binw
        inbin = np.logical_and(HTmag >= rlo, HTmag < rhi)
        reclPC[i] = np.sum(completePC[inbin]) / float(np.sum(inbin))
        reclCC[i] = np.sum(confmatchCC[inbin]) / float(np.sum(inbin))
        for a in xrange(len(gdat.catlrefr) - 1):
            reclrefr[a,i] = np.sum(cmplrefr[a][inbin]) / float(np.sum(inbin))
    
    plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclPC, c='r', label='Catalog Ensemble', marker='x', markersize=10, mew=2)
    plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclCC, c='purple', label='Condensed Catalog', marker='1', markersize=10, mew=2)
    for a in xrange(len(gdat.catlrefr) - 1):
        plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclrefr[a], c=gdat.colrrefr[a+1], label=gdat.lablrefr[a+1], marker='+', markersize=10, mew=2)
    plt.xlabel(gdat.lablrefr[0] + ' magnitude')
    plt.ylabel('completeness')
    plt.ylim((-0.1, 1.1))
    plt.legend(prop={'size':12}, loc = 'best')
    plt.tight_layout()
    plt.savefig(gdat.pathdatartag + 'cmpl.pdf')
    plt.close()
    
    nbins = 4
    nsigf = 71
    minr, maxr = 17.5, 21.5
    binw = (maxr - minr) / float(nbins)
    sigfaccuts = np.logspace(0, 1.4, num=nsigf)
    
    #table
    reclCCt = np.zeros((nbins, nsigf))
    precCCt = np.zeros((nbins, nsigf))
    
    for i in xrange(nbins):
        rlo = minr + i * binw
        rhi = rlo + binw
        inbinHT = np.logical_and(HTmag >= rlo, HTmag < rhi)
        inbinCC = np.logical_and(CCmag >= rlo, CCmag < rhi)
        for j in xrange(nsigf):
            sigfaccut = sigfaccuts[j]
            reclCCt[i,j] = np.sum(confmatchCC[np.logical_and(inbinHT, sigfCC < sigfaccut)]) / float(np.sum(inbinHT))
            precCCt[i,j] = np.sum(CCconf[np.logical_and(inbinCC, np.logical_and(goodmatchCC, CCs < sigfaccut))]) / float(np.sum(CCconf[np.logical_and(inbinCC, CCs < sigfaccut)]))
    
    
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    markers = ['o', 'v', 's', 'D', 'p', '*']
    #markers = ['$1.8$', '$3.2$', '$5.6$', '$10$', '$18$']
    
    for i in xrange(nbins):
        rlo = minr + i * binw
        rme = rlo + 0.5*binw
        rhi = rlo + binw
        
        if i == 0:
            label1 = 'DAOPHOT Catalog mag %2.0f' % rme
            label2 = 'Condensed Catalog mag %2.0f' % rme
        else:
            label1 = None
            label2 = 'mag %2.0f' % rme
        # make it convex: more stringent cuts that worsen precision are not used
        for j in xrange(nsigf-2, -1, -1):
            if precCCt[i,j] < precCCt[i,j+1]:
                precCCt[i,j] = precCCt[i,j+1]
                reclCCt[i,j] = reclCCt[i,j+1]
        
        # repeats make it maximally convex as allowed by data points
        plt.plot(1-np.repeat(precCCt[i,:], 2)[:-1], np.repeat(reclCCt[i,:], 2)[1:], c='purple', ls=linestyles[i], label=label2, zorder=2)
    
    for i in xrange(nbins):
        for j in xrange(len(markers)):
            if i == 0:
                if j == 0:
                    label3 = 'DF = %2.1f' % sigfaccuts[10*j+10]
                else:
                    label3 = '%2.1f' % sigfaccuts[10*j+10]
            else:
                label3 = None
            plt.scatter(1-precCCt[i,10*j+10], reclCCt[i,10*j+10], c='purple', marker=markers[j], s=100, label=label3, zorder=2)
    plt.xlabel('false discovery rate')
    plt.ylabel('completeness')
    plt.xlim((-0.1,1.1))
    plt.ylim((-0.1,1.1))
    plt.tight_layout()
    plt.savefig(gdat.pathdatartag + 'rofc%02i.%s' % (a, gdat.strgplotfile))
    plt.close()
    

def associate(a, mags_a, b, mags_b, dr, dmag, confs_b = None, sigfs_b = None):
    allmatches = a.query_ball_tree(b, dr)
    
    goodmatch = np.zeros(mags_a.size, np.bool)
    if confs_b is not None:
        confmatch = np.zeros(mags_a.size)
    if sigfs_b is not None:
        sigfmatch = np.zeros(mags_a.size) + float('inf')
    
    for i in xrange(len(allmatches)):
        matches = allmatches[i]
        if len(matches):
            mag_a = mags_a[i]
            goodmatch[i] = False
            for j in matches:
                mag_b = mags_b[j]
                if np.abs(mag_a - mag_b) < dmag:
                    goodmatch[i] = True
                    if (confs_b is not None) and (confs_b[j] > confmatch[i]):
                        confmatch[i] = confs_b[j]
                    if (sigfs_b is not None) and (sigfs_b[j] < sigfmatch[i]):
                        sigfmatch[i] = sigfs_b[j]

    if confs_b is not None:
        if sigfs_b is not None:
            return goodmatch, matches, confmatch, sigfmatch
        else:
            return goodmatch, matches, confmatch
    else:
        if sigfs_b is not None:
            return goodmatch, matches, sigfmatch
        else:
            return goodmatch, matches


def eval_modl(gdat, x, y, f, cntpback, offsxpos=0, offsypos=0, weig=None, cntprefr=None, clib=None, sizeimag=None):
        
    if gdat.verbtype > 1:
        print 'eval_modl()'
    
    if gdat.boolspre:
        assert x.dtype == np.float32
        assert y.dtype == np.float32
        assert f.dtype == np.float32
        #assert coefspix.dtype == np.float32
    if cntprefr is not None:
        if gdat.boolspre:
            assert cntprefr.dtype == np.float32
        boolretrdoub = True
    else:
        boolretrdoub = False
    
    # define image and region size for PSF plotting
    if sizeimag is None:
        sizeimag = gdat.sizeimag
        sizeregi = gdat.sizeregi
    else:
        sizeregi = max(sizeimag[0], sizeimag[1])
    
    if weig is None:
        if gdat.boolspre:
            weig = np.ones([gdat.numbener] + sizeimag +[gdat.numbtime], dtype=np.float32)
        else:
            weig = np.ones([gdat.numbener] + sizeimag +[gdat.numbtime])
    
    if gdat.diagmode:
        pass
        #if x.size < 1 or y.size < 1 or f.size < 1:
        #    raise Exception('')
    
    if gdat.verbtype > 2:
        print 'sizeimag'
        print sizeimag
        print 'x y'
        for xt, yt in zip(x, y):
            print xt, yt

    # temp -- sometimes phonions are outside image... what is best way to handle?
    if clib is not None:
        goodsrc = (x > 0) * (x < sizeimag[0] - 1) * (y > 0) * (y < sizeimag[1] - 1)
        x = x.compress(goodsrc)
        y = y.compress(goodsrc)
        f = f.compress(goodsrc, axis=2)
    
    if gdat.verbtype > 1:
        print 'x'
        summgene(x)
        print 'y'
        summgene(y)
        print 'f'
        summgene(f)
        print

    numbphon = x.size
    rad = gdat.numbsidepsfn / 2
    
    if gdat.verbtype > 2:
        print 'After compressing:'
        print 'x y'
        for xt, yt in zip(x, y):
            print xt, yt

    ix = np.rint(x).astype(np.int32)
    dx = ix - x
    iy = np.rint(y).astype(np.int32)
    dy = iy - y
     
    #print 'heeey'
    #print 'x'
    #print x
    #print 'y'
    #print y
    #print 'ix'
    #print ix
    #print 'iy'
    #print iy
    #print 'dx'
    #print dx
    #print 'dy'
    #print dy
    
    # construct the design matrix
    if gdat.boolspre:
        desimatr = np.column_stack((np.full(numbphon, 1., dtype=np.float32), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, \
                                                                                    dx*dx*dy, dx*dy*dy, dy*dy*dy)).astype(np.float32)[None, None, :, :] * f[:, :, :, None]
    else:
        desimatr = np.column_stack((np.full(numbphon, 1.), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, \
                                                                                    dx*dx*dy, dx*dy*dy, dy*dy*dy))[None, None, :, :] * f[:, :, :, None]

    chi2 = np.zeros((gdat.numbregiyaxi, gdat.numbregixaxi), dtype=np.float64)
    
    if gdat.boolspre:
        cntpmodl = np.ones([gdat.numbener] + sizeimag +[gdat.numbtime], dtype=np.float32)
    else: 
        cntpmodl = np.ones([gdat.numbener] + sizeimag +[gdat.numbtime])
    
    if cntprefr is None:
        if gdat.boolspre:
            cntprefr = np.zeros([gdat.numbener] + sizeimag +[gdat.numbtime], dtype=np.float32)
        else:
            cntprefr = np.zeros([gdat.numbener] + sizeimag +[gdat.numbtime])
    
    if clib is None:
        
        if gdat.verbtype > 1:
            print 'Not using the C library...'
    
        
        if gdat.verbtype > 2:
            print 'gdat.coefspix'
            summgene(gdat.coefspix)
            print 'numbphon'
            print numbphon
            print 'gdat.numbsidepsfn'
            print gdat.numbsidepsfn
        for i in gdat.indxener:
            for t in gdat.indxtime:
                chi2temp = np.zeros((gdat.numbregiyaxi, gdat.numbregixaxi), dtype=np.float64)
                
                cntpmodltemp = np.full((sizeimag[1]+2*rad+1,sizeimag[0]+2*rad+1), 0., dtype=np.float32)
                cntpmodltemp[rad:sizeimag[1]+rad, rad:sizeimag[0]+rad] = np.copy(cntpback[i, :, :, t])
                
                #if gdat.boolspre:
                #    cntpmodltemp = np.full((sizeimag[1]+2*rad+1,sizeimag[0]+2*rad+1), cntpback, dtype=np.float32)
                #else: 
                #    cntpmodltemp = np.full((sizeimag[1]+2*rad+1,sizeimag[0]+2*rad+1), cntpback)
            
                if gdat.boolbili:
                    # temp -- make this compatible with float32
                    cntpmodlstmp2 = np.empty((numbphon, gdat.numbsidepsfn, gdat.numbsidepsfn))
                    for k in range(numbphon):
                        #print 'x[k]'
                        #print x[k]
                        #print 'y[k]'
                        #print y[k]
                        #print 'ix[k]'
                        #print ix[k]
                        #print 'iy[k]'
                        #print iy[k]
                        #print 'Calling retr_psfnbili()'
                        cntpmodlstmp2[k, :, :] = retr_psfnbili(gdat, i, x[k] - ix[k], y[k] - iy[k]) * f[i, t, k]
                else:
                    cntpmodlstmp2 = np.dot(desimatr[i, t, :, :], gdat.coefspix).reshape((numbphon, gdat.numbsidepsfn, gdat.numbsidepsfn))
                    
                cntpmodlstmp = np.zeros((numbphon,gdat.numbsidepsfn,gdat.numbsidepsfn), dtype=np.float32)
                cntpmodlstmp[:,:,:] = cntpmodlstmp2[:,:,:]
                for k in xrange(numbphon):
                    #print 'k'
                    #print k
                    #print 'gdat.numbsidepsfn'
                    #print gdat.numbsidepsfn
                    #print 'rad'
                    #print rad
                    #print 'iy'
                    #print iy
                    #print 'cntpmodltemp[iy[k]:iy[k]+rad+rad+1,ix[k]:ix[k]+rad+rad+1]'
                    #summgene(cntpmodltemp[iy[k]:iy[k]+rad+rad+1,ix[k]:ix[k]+rad+rad+1])
                    #print 'cntpmodlstmp[k, :, :]'
                    #summgene(cntpmodlstmp[k, :, :])
                    #print

                    cntpmodltemp[iy[k]:iy[k]+rad+rad+1,ix[k]:ix[k]+rad+rad+1] += cntpmodlstmp[k, :, :]
                cntpmodl[i, :, :, t] = cntpmodltemp[rad:sizeimag[1]+rad,rad:sizeimag[0]+rad]
                
                if gdat.diagmode:
                    if not np.isfinite(cntpmodltemp).all():
                        raise Exception('')
                    if not np.isfinite(cntpmodlstmp2).all():
                        raise Exception('')
                    if not np.isfinite(cntpmodl).all():
                        raise Exception('')

                if cntprefr is not None:
                    cntpdiff = cntprefr[i, :, :, t] - cntpmodl[i, :, :, t]
                
                if gdat.verbtype > 1:
                    print 'cntpdiff'
                    summgene(cntpdiff)
                    print 'weig'
                    summgene(weig)
                
                for v in xrange(gdat.numbregiyaxi):
                    y0 = max(v*sizeregi - offsypos - gdat.marg, 0)
                    y1 = min((v+1)*sizeregi - offsypos + gdat.marg, sizeimag[1])
                    for u in xrange(gdat.numbregixaxi):
                        x0 = max(u*sizeregi - offsxpos - gdat.marg, 0)
                        x1 = min((u+1)*sizeregi - offsxpos + gdat.marg, sizeimag[0])
                        cntpdifftemp = cntpdiff[y0:y1,x0:x1]
                        chi2temp[v, u] = np.sum(cntpdifftemp**2 * weig[i, y0:y1,x0:x1, t])
                
                chi2 += chi2temp
                
                #print 'it'
                #print i, t
                #print 'chi2temp'
                #print chi2temp
                #print 'np.sum(weig[i, :, :, t] * (cntpdiff)**2)'
                #print np.sum(weig[i, :, :, t] * (cntpdiff)**2)
        #print 'chi2'
        #print chi2

    else:
        
        if gdat.verbtype > 1:
            print 'Using the C library...'
    
        # counts per pixel of model evaluated over postage stamps
        if gdat.boolspre:
            cntpmodlstmp = np.zeros((numbphon, gdat.numbpixlpsfn), dtype=np.float32)
        else:
            cntpmodlstmp = np.zeros((numbphon, gdat.numbpixlpsfn))
        
        if gdat.verbtype > 2:
            print 'f'
            summgene(f)
            print 'gdat.numbregiyaxi'
            print gdat.numbregiyaxi
            print 'gdat.numbregixaxi'
            print gdat.numbregixaxi
            print 'gdat.coefspix'
            summgene(gdat.coefspix)
            print 'iy'
            summgene(iy)
            print 'cntpmodl'
            summgene(cntpmodl)
            print 'cntprefr'
            summgene(cntprefr)
            print 'weig'
            print weig.shape
            print 'chi2'
            print chi2.shape

        chi2temp = np.empty((gdat.numbregiyaxi, gdat.numbregixaxi), dtype=np.float64)
        for i in gdat.indxener:
            for t in gdat.indxtime:
                
                cntpmodltemp = np.copy(cntpback[i, :, :, t])
                if gdat.boolspre:
                    cntpmodltemp = cntpmodltemp.astype(np.float32)

                #if gdat.boolspre:
                #    cntpmodltemp = np.zeros(sizeimag, dtype=np.float32) + cntpback
                #else:
                #    cntpmodltemp = np.zeros(sizeimag) + cntpback
                
                #cntprefrtemp = cntprefr[i, :, :, t]
                cntprefrtemp = np.ascontiguousarray(cntprefr[i, :, :, t])

                #weigtemp = weig[i, :, :, t]
                weigtemp = np.ascontiguousarray(weig[i, :, :, t])
                
                if gdat.verbtype > 2:
                    print 'it'
                    print i, t
                    print 'sizeimag'
                    print sizeimag
                    print 'numbphon'
                    print numbphon
                    print 'gdat.numbsidepsfn'
                    print gdat.numbsidepsfn
                    print 'gdat.numbparaspix'
                    print gdat.numbparaspix
                    print 'desimatr[i, t, :, :]'
                    summgene(desimatr[i, t, :, :])
                    print 'gdat.coefspix[i, :, :]'
                    summgene(gdat.coefspix[i, :, :])
                    print 'cntpmodlstmp'
                    summgene(cntpmodlstmp)
                    print 'ix'
                    print ix
                    print 'iy'
                    print iy
                    print 'cntpmodltemp'
                    summgene(cntpmodltemp)
                    print 'cntprefrtemp'
                    summgene(cntprefrtemp)
                    print 'weigtemp'
                    summgene(weigtemp)
                    print 'sizeregi'
                    print sizeregi
                    print 'gdat.marg'
                    print gdat.marg
                    print 'offsxpos'
                    print offsxpos
                    print 'offsypos'
                    print offsypos
                    print 'gdat.booltile'
                    print gdat.booltile
                    print 'desimatr[i, t, :, :].flags[C_CONTIGUOUS]'
                    print desimatr[i, t, :, :].flags['C_CONTIGUOUS']
                    print 'gdat.coefspix[i, :, :].flags[C_CONTIGUOUS]'
                    print gdat.coefspix[i, :, :].flags['C_CONTIGUOUS']
                    print 'cntpmodlstmp.flags[C_CONTIGUOUS]'
                    print cntpmodlstmp.flags['C_CONTIGUOUS']
                    print 'cntpmodltemp.flags[C_CONTIGUOUS]'
                    print cntpmodltemp.flags['C_CONTIGUOUS']
                    print 'cntprefrtemp.flags[C_CONTIGUOUS]'
                    print cntprefrtemp.flags['C_CONTIGUOUS']
                    print 'weigtemp.flags[C_CONTIGUOUS]'
                    print weigtemp.flags['C_CONTIGUOUS']
                    print 'chi2temp.flags[C_CONTIGUOUS]'
                    print chi2temp.flags['C_CONTIGUOUS']

                clib(sizeimag[0], sizeimag[1], numbphon, gdat.numbsidepsfn, gdat.numbparaspix, desimatr[i, t, :, :], \
                                                               gdat.coefspix[i, :, :], cntpmodlstmp, ix, iy, cntpmodltemp, \
                                                               cntprefrtemp, weigtemp, chi2temp, sizeregi, gdat.marg, offsxpos, offsypos, gdat.booltile)
               
                # temp
                #chi2temp = np.array([[np.sum((cntprefrtemp - cntpmodltemp)**2 * weig[i, :, :, t])]]).astype(np.float64)
                
                if gdat.verbtype > 1:
                    print 'chi2temp'
                    summgene(chi2temp)
                    print 'cntpmodltemp'
                    summgene(cntpmodltemp)
                    print
                
                if gdat.diagmode:
                    if not np.isfinite(chi2temp).all():
                        raise Exception('')

                    if not np.isfinite(cntpmodltemp).all():
                        raise Exception('')

                chi2 += chi2temp
                cntpmodl[i, :, :, t] = cntpmodltemp
                
                #print 'it'
                #print i, t
                #print 'np.sum((cntprefrtemp - cntpmodl[i, :, :, t])**2 * weig[i, :, :, t])'
                #print np.sum((cntprefrtemp - cntpmodl[i, :, :, t])**2 * weig[i, :, :, t])
                #print 'chi2temp'
                #print chi2temp

    #print 'Exiting eval_modl()'
    
    if gdat.diagmode:
        if not np.isfinite(cntpmodl).all():
            raise Exception('')

    if gdat.verbtype > 2:
        print 'cntpmodl'
        summgene(cntpmodl)
        print 'eval_modl() ended'

    if boolretrdoub:
        return cntpmodl, chi2
    else:
        return cntpmodl


def retr_coefspix(gdat):
   
    # pad by one row and one column on each side
    if gdat.boolspre:
        gdat.cntppsfnusampadd = np.zeros((gdat.numbener, gdat.numbsidepsfnusam+2, gdat.numbsidepsfnusam+2), dtype=np.float32)
    else:
        gdat.cntppsfnusampadd = np.zeros((gdat.numbener, gdat.numbsidepsfnusam+2, gdat.numbsidepsfnusam+2))
    gdat.cntppsfnusampadd[:, 1:gdat.numbsidepsfnusam+1, 1:gdat.numbsidepsfnusam+1] = gdat.cntppsfnusam

    # make design matrix for each original pixel
    gdat.numbsidepsfn = gdat.numbsidepsfnusam / gdat.factusam # dimension of original psf
    assert(gdat.numbsidepsfn % 2 == 1)
    rad = gdat.factusam / 2 # upsampling factor / 2, rounded down
    # eg, factusam = 4 -> rad 2 -> subpixel shifts [-3/4, -1/2, -1/4, 0, 1/4, 1/2, 3/4]
    #     factusam = 5 -> rad 2 -> subpixel shifts [-3/5, -2/5, -1/5, 0, 1/5, 2/5, 3/5]
    nx = 2*rad + 3
    y, x = np.mgrid[-rad-1:rad+2, -rad-1:rad+2] / np.float32(gdat.factusam)
    x = x.flatten()
    y = y.flatten()
    if gdat.boolspre:
        A = np.column_stack([np.full(nx*nx, 1, dtype=np.float32), x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y]).astype(np.float32)
    else:
        A = np.column_stack([np.full(nx*nx, 1), x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y])
    
    # number of subpixel parameters
    gdat.numbparaspix = A.shape[1]

    # output array of coefficients
    if gdat.boolspre:
        gdat.coefspix = np.zeros((gdat.numbener, gdat.numbparaspix, gdat.numbsidepsfn, gdat.numbsidepsfn), dtype=np.float32)
    else:
        gdat.coefspix = np.zeros((gdat.numbener, gdat.numbparaspix, gdat.numbsidepsfn, gdat.numbsidepsfn))

    # loop over original psf pixels and get fit coefficients
    for i in gdat.indxener:
        for a in xrange(gdat.numbsidepsfn):
            for j in xrange(gdat.numbsidepsfn):
                # solve p = A gdat.coefspix for gdat.coefspix
                p = gdat.cntppsfnusampadd[i, a*gdat.factusam:a*gdat.factusam+nx, j*gdat.factusam:j*gdat.factusam+nx].flatten()
                gdat.coefspix[i, :, a, j] = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, p)) 
        
        if gdat.diagmode:
            if np.amax(gdat.coefspix[i, :, :, :]) == 0.:
                raise Exception('')

    gdat.coefspix = gdat.coefspix.reshape(gdat.numbener, gdat.numbparaspix, gdat.numbsidepsfn**2)
    
    if gdat.boolplotsave:
        for i in gdat.indxener:
            figr, axis = plt.subplots()
            axis.imshow(gdat.cntppsfnusampadd[i, :, :], interpolation='nearest')
            plt.tight_layout()
            plt.savefig(gdat.pathdatartag + '%s_psfnusamp%04d.%s' % (gdat.rtag, i, gdat.strgplotfile))
            plt.close()


def neighbours(x,y,neigh,i,generate=False):
    neighx = np.abs(x - x[i])
    neighy = np.abs(y - y[i])
    adjacency = np.exp(-(neighx*neighx + neighy*neighy)/(2.*neigh*neigh))
    adjacency = adjacency.astype(np.float64) # otherwise underflows cause problems
    oldadj = adjacency.copy()
    adjacency[adjacency < 4e-6] = 0 # reject > 5 neigh pairs
    adjacency[i] = 0.
    neighbours = np.sum(adjacency)
    if generate:
        if neighbours:
            # temp
            #print 'adjacency'
            #summgene(adjacency)
            #print 'neighbours'
            '''print neighbours
            np.set_printoptions(precision=15)
            print 'THIS SHOULD BE ZERO'
            adjacency = adjacency.astype(np.float64)
            print np.sum(adjacency / np.sum(adjacency)) - 1

            print 'this is for funzies'
            lols = np.zeros(adjacency.size) + 1./float(adjacency.size) 
            print np.random.choice(adjacency.size, p=lols)'''

            adjacency /= np.sum(adjacency)
            j = np.random.choice(adjacency.size, p=adjacency.flatten())
        else:
            j = -1
        return neighbours, j
    else:
        return neighbours


def get_region(x, offsxpos, sizeregi):
    
    return np.floor(x + offsxpos).astype(np.int) / sizeregi


# visualization-related functions
def setp_imaglimt(gdat, axis):
    
    axis.set_xlim(-0.5, gdat.sizeimag[0] - 0.5)
    axis.set_ylim(-0.5, gdat.sizeimag[1] - 0.5)
                    
    
def idx_parity(x, y, n, offsxpos, offsypos, parity_x, parity_y, sizeregi):
    
    match_x = (get_region(x[0:n], offsxpos, sizeregi) % 2) == parity_x
    match_y = (get_region(y[0:n], offsypos, sizeregi) % 2) == parity_y
    
    #print 'offsxpos'
    #print offsxpos
    #print 'sizeregi'
    #print sizeregi
    #print 'parity_x'
    #print parity_x
    #print 'get_region(x[0:n], offsxpos, sizeregi)'
    #print get_region(x[0:n], offsxpos, sizeregi)
    #print 'match_x'
    #print match_x
    
    return np.flatnonzero(np.logical_and(match_x, match_y))


def retr_numbpara(gdat, numbstar, numbgalx):
    
    numbpara = (2 + gdat.numbflux) * numbstar + (5 + gdat.numbflux) * numbgalx

    return numbpara


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


def writ_catl(gdat, catl, path):
    
    filetemp = h5py.File(path, 'w')
    print 'Writing to %s...' % path
    filetemp.create_dataset('numb', data=catl['numb'])
    for strgfeat in gdat.liststrgfeatstar:
        if strgfeat == 'flux' and 'catlseed' in path:
            continue
        filetemp.create_dataset(strgfeat, data=catl[strgfeat])
    if 'catlseed' in path:
        filetemp.create_dataset('degr', data=catl['degr'])
    if catl['xpos'].ndim == 2:
        filetemp.create_dataset('chi2', data=catl['chi2'])
        filetemp.create_dataset('lposterm', data=catl['lposterm'])
        
    filetemp.close()


def read_catl(gdat, path, boolmock=False):
    
    filetemp = h5py.File(path, 'r')
    print 'Reading %s...' % path
    catl = {}
    catl['numb'] = filetemp['numb'][()]
    for strgfeat in gdat.liststrgfeatstar:
        if strgfeat == 'flux' and 'catlseed' in path:
            continue
        catl[strgfeat] = filetemp[strgfeat][()] 
    if boolmock:
        catl['apho'] = filetemp['apho'][()] 
        catl['strg'] = filetemp['strg'][()] 
    if 'catlseed' in path:
        catl['degr'] = filetemp['degr'][()] 
    if catl['xpos'].ndim == 2:
        catl['chi2'] = filetemp['chi2'][()] 
        catl['lposterm'] = filetemp['lposterm'][()] 
    filetemp.close()
   
    return catl


def setp(gdat):
    
    gdat.sizeimag = [gdat.numbsidexpos, gdat.numbsideypos]
    
    if gdat.sizeregi is None:
        if gdat.booltile and gdat.sizeimag[0] > 30:
            gdat.sizeregi = gdat.sizeimag[0] / 2
        else:
            gdat.sizeregi = gdat.sizeimag[0]
    
    if gdat.sizeimag[0] % gdat.sizeregi != 0 or gdat.sizeimag[1] % gdat.sizeregi != 0:
        print 'gdat.sizeimag'
        print gdat.sizeimag
        print 'gdat.sizeregi'
        print gdat.sizeregi
        raise Exception('')
        
    if gdat.booltile:
        gdat.numbregiyaxi = gdat.sizeimag[1] / gdat.sizeregi + 1
        gdat.numbregixaxi = gdat.sizeimag[0] / gdat.sizeregi + 1
    else:
        gdat.numbregiyaxi = 1
        gdat.numbregixaxi = 1
    
    gdat.liststrgfeatstar = ['xpos', 'ypos', 'flux']
    gdat.liststrgparastar = list(gdat.liststrgfeatstar)
    gdat.liststrgparastar += ['numb']
    gdat.liststrgfeatgalx = ['xpos', 'ypos', 'flux', 'xxmo', 'xymo', 'yymo']
    gdat.liststrgparagalx = list(gdat.liststrgfeatgalx)
    gdat.liststrgparagalx += ['numbgalx']


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


def plot_psfn(gdat, k, ix, iy, clib=None):
    
    print 'Plotting the PSF...'
    
    xpos = np.array([gdat.numbsidepsfn / 2 + ix])
    ypos = np.array([gdat.numbsidepsfn / 2 + iy])
    flux = np.ones((gdat.numbener, gdat.numbtime, 1))
    if gdat.boolspre:
        xpos = xpos.astype(np.float32)
        ypos = ypos.astype(np.float32)
        flux = flux.astype(np.float32)
    cntpback = np.zeros((gdat.numbener, gdat.numbsidepsfn, gdat.numbsidepsfn, gdat.numbtime))
    sizeimag = [gdat.numbsidepsfn, gdat.numbsidepsfn]
    
    cntpmodlpsfn = eval_modl(gdat, xpos, ypos, flux, cntpback, clib=clib, sizeimag=sizeimag)
    
    if gdat.exprtype == 'tess':
        vmax = 0.5
    else:
        vmax = None
    #print 'ix'
    #print ix
    #print 'iy'
    #print iy
    #np.set_printoptions(1)
    for i in gdat.indxener:
        
        figr, axis = plt.subplots(figsize=(20, 20))
        imag = axis.imshow(cntpmodlpsfn[i, :, :, 0], interpolation='nearest', origin='lower', cmap='Greys_r', vmin=0, vmax=vmax)
        axis.scatter(gdat.numbsidepsfn/2, gdat.numbsidepsfn/2, marker='x', s=100, color='b')
        axis.scatter(xpos, ypos, marker='+', s=100, color='g')
        axis.set_title('x,y = %.3g %.3g' % (xpos, ypos)) 
        axis.set_position([0.1, 0.1, 0.7, 0.7], which='both')
        cbaxes = figr.add_axes([0.85, 0.1, 0.05, 0.7]) 
        cbar = figr.colorbar(imag, cax=cbaxes) 
        plt.savefig(gdat.pathdatartag + '%s_psfnspix%04d_%04d.%s' % (gdat.rtag, i, k, gdat.strgplotfile))
        plt.close()

        psfnbili = retr_psfnbili(gdat, i, xpos, ypos)

        figr, axis = plt.subplots(figsize=(20, 20))
        imag = axis.imshow(psfnbili, interpolation='nearest', origin='lower', cmap='Greys_r')
        axis.scatter(gdat.numbsidepsfn/2, gdat.numbsidepsfn/2, marker='x', s=100, color='b')
        axis.scatter(xpos, ypos, marker='+', s=100, color='g')
        axis.set_position([0.1, 0.1, 0.7, 0.7], which='both')
        cbaxes = figr.add_axes([0.85, 0.1, 0.05, 0.7]) 
        cbar = figr.colorbar(imag, cax=cbaxes) 
        plt.savefig(gdat.pathdatartag + '%s_psfnbili%04d_%04d.%s' % (gdat.rtag, i, k, gdat.strgplotfile))
        plt.close()
        
        psfnbiliinve = np.zeros(sizeimag)
        mask = psfnbili > 0
        psfnbiliinve[mask] = 1. / psfnbili[mask]
        figr, axis = plt.subplots(figsize=(20, 20))
        imag = axis.imshow((cntpmodlpsfn[i, :, :, 0] - psfnbili) * psfnbiliinve, interpolation='nearest', origin='lower', cmap=gdat.cmapresi)
        axis.scatter(gdat.numbsidepsfn/2, gdat.numbsidepsfn/2, marker='x', s=100, color='b')
        axis.scatter(xpos, ypos, marker='+', s=100, color='g')
        axis.set_position([0.1, 0.1, 0.7, 0.7], which='both')
        cbaxes = figr.add_axes([0.85, 0.1, 0.05, 0.7]) 
        cbar = figr.colorbar(imag, cax=cbaxes) 
        path = gdat.pathdatartag + '%s_psfnbilifres%04d_%04d.%s' % (gdat.rtag, i, k, gdat.strgplotfile)
        print 'Writing to %s...' % path
        plt.savefig(path)
        plt.close()


def retr_psfnbili(gdat, i, xpos, ypos):
    
    iix = int(np.floor((xpos + 0.5) * gdat.factusam))
    iiy = int(np.floor((ypos + 0.5) * gdat.factusam))
    dix = (xpos + 0.5) * gdat.factusam - iix
    diy = (ypos + 0.5) * gdat.factusam - iiy
    
    iix = iix % gdat.factusam
    iiy = iiy % gdat.factusam
    
    #print 'retr_psfnbili'
    #print 'xpos'
    #print xpos
    #print 'ypos'
    #print ypos
    #print 'iix'
    #print iix
    #print 'iiy'
    #print iiy
    #print 'dix'
    #print dix
    #print 'diy'
    #print diy
    
    f00 = gdat.cntppsfnusam[i, iiy-1:gdat.numbsidepsfnusam:gdat.factusam,   iix-1:gdat.numbsidepsfnusam:gdat.factusam]
    f01 = gdat.cntppsfnusam[i, iiy:gdat.numbsidepsfnusam:gdat.factusam, iix-1:gdat.numbsidepsfnusam:gdat.factusam]
    f10 = gdat.cntppsfnusam[i, iiy-1:gdat.numbsidepsfnusam:gdat.factusam,   iix:gdat.numbsidepsfnusam:gdat.factusam]
    f11 = gdat.cntppsfnusam[i, iiy:gdat.numbsidepsfnusam:gdat.factusam, iix:gdat.numbsidepsfnusam:gdat.factusam]
    psfnbili = f00 * (1. - dix) * (1. - diy) + f10 * dix * (1. - diy) + f01 * (1. - dix) * diy + f11 * dix * diy
    
    #print 'f00'
    #print f00
    #print 'f01'
    #print f01
    #print 'f10'
    #print f10
    #print 'f11'
    #print f11
    #print 'psfnbili'
    #print psfnbili
    #print 

    return psfnbili


def plot_lcur(gdat, boolsimp=False):
    
    listplottype = ['nomi', 'frac']
    if gdat.datatype == 'mock':
        listplottype += ['resi']
    
    for plottype in listplottype:
        cntr = 0
        for k in gdat.indxsourcond:
            print 'boolsimp'
            print boolsimp
            print 'plottype'
            print plottype
            print 'k'
            print k
            
            numblcurplot = min(4, gdat.numbsourcond - k)
            
            meanlcur = np.mean(gdat.catlcond[gdat.indxflux, k, 0], axis=0)
            if not boolsimp:
                errrlcur = np.empty((2, gdat.numbtime))
                for i in range(2):
                    errrlcur[i, :] = np.mean(gdat.catlcond[gdat.indxflux, k, 1], axis=0)
            
            if (gdat.catlcond[gdat.indxflux, k, 1] == 0.).any():
                print 'Warning! Condensed catalog flux uncertainties vanish!'
                print 'gdat.catlcond[gdat.indxflux, k, 1]'
                print gdat.catlcond[gdat.indxflux, k, 1]
                print 'gdat.catlcond[gdat.indxflux, k, 0]'
                print gdat.catlcond[gdat.indxflux, k, 0]
                #raise Exception('')

            if k % numblcurplot == 0:
                figr, axis = plt.subplots(numblcurplot, 1, figsize=(12, 4 * numblcurplot))
                if numblcurplot == 1:
                    axis = [axis]
            if len(gdat.catlrefr) > 0 and not boolsimp and plottype != 'frac':
                indxtrue = np.where(gdat.indxmtch == k)[0]
                if indxtrue.size > 0:
                    if indxtrue.size > 1:
                        print 'Warning! indxtrue size is larger than 1'
                        print 'indxtrue'
                        print indxtrue
                    indxtrue = indxtrue[0]
                    meanlcurtrue = np.mean(gdat.catlrefr[0]['flux'][:, :, indxtrue], axis=0)
                    if plottype == 'nomi':
                        axis[k%numblcurplot].plot(gdat.indxtime, meanlcurtrue, color='g', markersize=10, marker='o', alpha=0.5)
            
                    if gdat.datatype == 'mock':
                        ydattemp = np.mean(gdat.catlrefr[0]['apho'][:, :, indxtrue], axis=0)
                        if plottype == 'nomi' and not boolsimp:
                            print 'Aperture RMS'
                            print np.sqrt(np.mean((ydattemp - np.mean(ydattemp))**2))
                            print 'np.mean(ydattemp)'
                            print np.mean(ydattemp)
                        ydattemp /= np.mean(ydattemp)
                        ydattemp *= np.mean(meanlcur)
                        if plottype == 'resi':
                            ydattemptemp = ydattemp - np.mean(ydattemp)
                            ydattemp = 1e6 * ydattemptemp / np.mean(ydattemp)
                        axis[k%numblcurplot].plot(gdat.indxtime, ydattemp, color='m', markersize=10, marker='o', alpha=0.5)
                
                    if plottype == 'resi':
                        meanlcur -= meanlcurtrue
                        meanlcur /= meanlcurtrue
                        meanlcur *= 1e6
                        errrlcur /= meanlcurtrue
                        errrlcur *= 1e6
           
            if plottype == 'frac':
                meanlcurtemp = meanlcur - np.mean(meanlcur)
                meanlcur = meanlcurtemp / meanlcur
                meanlcur *= 1e6
                if not boolsimp:
                    errrlcur /= np.mean(meanlcur)
                    errrlcur *= 1e6
            
            axis[k%numblcurplot].set_xlim([-1, gdat.numbtime])
            
            if boolsimp:
                axis[k%numblcurplot].plot(gdat.indxtime, meanlcur, color='b', marker='o', ls='', markersize=10, alpha=1.)
            else:
                if plottype == 'nomi' and not boolsimp:
                    print 'PCAT RMS'
                    print np.sqrt(np.mean((meanlcur - np.mean(meanlcur))**2))
                    print 'np.mean(meanlcur)'
                    print np.mean(meanlcur)
                temp, listcaps, temp = axis[k%numblcurplot].errorbar(gdat.indxtime, meanlcur, yerr=errrlcur, color='b', marker='o', ls='', markersize=10, alpha=0.5)
                for caps in listcaps:
                    caps.set_markeredgewidth(10)
            
            if plottype == 'frac' or plottype == 'resi':
                ylim = axis[k%numblcurplot].get_ylim()
                maxmylim = max(abs(ylim[0]), abs(ylim[1]))
                ylim = [-maxmylim, maxmylim]
                axis[k%numblcurplot].set_ylim(ylim)
           
            if plottype == 'resi' or plottype =='frac':
                axis[k%numblcurplot].axhline(0., ls='--', color='grey', alpha=0.2)
                axis[k%numblcurplot].set_ylabel(r'$\sigma$ [ppm]')
            else:
                axis[k%numblcurplot].set_ylabel(r'$f$ [DN]')
    
            if plottype == 'nomi' and not boolsimp:
                axis[k%numblcurplot].text(0.85, 0.75, str(k), horizontalalignment='center', verticalalignment='center', transform=axis[k%numblcurplot].transAxes, color='b')
                if len(gdat.catlrefr) > 0 and indxtrue.size > 0:
                    axis[k%numblcurplot].text(0.95, 0.75, str(indxtrue), horizontalalignment='center', verticalalignment='center', \
                                                                                                                transform=axis[k%numblcurplot].transAxes, color='g')

            if (k + 1) % numblcurplot == 0 or k == gdat.numbsourcond - 1:
                try:
                    plt.tight_layout()
                except:
                    print 'Tight layout failed.'
                if boolsimp:
                    strgsimp = 'simp'
                else:
                    strgsimp = ''
                plt.savefig(gdat.pathdatartag + '%s_lcur%04d_%s%s.%s' % (gdat.rtag, cntr, plottype, strgsimp, gdat.strgplotfile))
                plt.close()
                cntr += 1


def setp_clib(gdat, gdatnotp, pathlion):
    
    if gdat.boolspre:
        array_2d_float    = npct.ndpointer(dtype=np.float32, ndim=2)
        array_1d_float    = npct.ndpointer(dtype=np.float32, ndim=1)
    array_1d_int      = npct.ndpointer(dtype=np.int32  , ndim=1)
    array_2d_double   = npct.ndpointer(dtype=np.float64, ndim=2)
    array_2d_int      = npct.ndpointer(dtype=np.int32,   ndim=2)
    
    #array_2d_float    = npct.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
    #array_1d_int      = npct.ndpointer(dtype=np.int32  , ndim=1, flags="C_CONTIGUOUS")
    #array_1d_float    = npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    #array_2d_double   = npct.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
    #array_2d_int      = npct.ndpointer(dtype=np.int32,   ndim=2, flags="C_CONTIGUOUS")
    
    gdatnotp.clib = npct.load_library(pathlion + 'blas', '.')
    gdatnotp.clib.clib_eval_modl.restype = None
    gdatnotp.clib.clib_updt_modl.restype = None
    gdatnotp.clib.clib_eval_llik.restype = None
    gdatnotp.clib.clib_eval_modl.argtypes = [c_int, c_int, c_int, c_int, c_int]
    gdatnotp.clib.clib_updt_modl.argtypes = [c_int, c_int]
    gdatnotp.clib.clib_eval_llik.argtypes = [c_int, c_int]
    if gdat.boolspre:
        gdatnotp.clib.clib_eval_modl.argtypes += [array_2d_float, array_2d_float, array_2d_float]
    else:
        gdatnotp.clib.clib_eval_modl.argtypes += [array_2d_double, array_2d_double, array_2d_double]
    gdatnotp.clib.clib_eval_modl.argtypes += [array_1d_int, array_1d_int]
    if gdat.boolspre:
        gdatnotp.clib.clib_eval_modl.argtypes += [array_2d_float, array_2d_float, array_2d_float]
        gdatnotp.clib.clib_updt_modl.argtypes += [array_2d_float, array_2d_float]
        gdatnotp.clib.clib_eval_llik.argtypes += [array_2d_float, array_2d_float, array_2d_float]
    else:
        gdatnotp.clib.clib_eval_modl.argtypes += [array_2d_double, array_2d_double, array_2d_double]
        gdatnotp.clib.clib_updt_modl.argtypes += [array_2d_double, array_2d_double]
        gdatnotp.clib.clib_eval_llik.argtypes += [array_2d_double, array_2d_double, array_2d_double]
    gdatnotp.clib.clib_eval_modl.argtypes += [array_2d_double]
    gdatnotp.clib.clib_eval_llik.argtypes += [array_2d_double]
    gdatnotp.clib.clib_eval_modl.argtypes += [c_int, c_int, c_int, c_int]
    gdatnotp.clib.clib_updt_modl.argtypes += [array_2d_int, c_int, c_int, c_int, c_int]
    gdatnotp.clib.clib_eval_llik.argtypes += [c_int, c_int, c_int, c_int]



def retr_axis(gdat, strgvarb, minm=None, maxm=None, numb=None, bins=None, scal='self', invr=False, strginit=''):
   
    if bins is None:
        if scal == 'self' or scal == 'pois' or scal == 'gaus':
            binsscal = np.linspace(minm, maxm, numb + 1)
        if scal == 'logt':
            binsscal = np.linspace(log10(minm), log10(maxm), numb + 1)
        if scal == 'asnh':
            binsscal = np.linspace(np.arcsinh(minm), np.arcsinh(maxm), numb + 1)
    
        if invr:
            binsscal = binsscal[::-1]
    
        meanvarbscal = (binsscal[1:] + binsscal[:-1]) / 2.
    
        if scal == 'self' or scal == 'pois' or scal == 'gaus':
            meanvarb = meanvarbscal
            bins = binsscal
        if scal == 'logt':
            meanvarb = 10**meanvarbscal
            bins = 10**binsscal
        if scal == 'asnh':
            meanvarb = np.sinh(meanvarbscal)
            bins = np.sinh(binsscal)
    else:
        numb = bins.size - 1

    indx = np.arange(numb)
    delt = np.diff(bins)
    limt = np.array([np.amin(bins), np.amax(bins)])

    setattr(gdat, strginit + 'limt' + strgvarb, limt)
    setattr(gdat, strginit + 'bins' + strgvarb, bins)
    setattr(gdat, strginit + 'mean' + strgvarb, meanvarb)
    setattr(gdat, strginit + 'delt' + strgvarb, delt)
    setattr(gdat, strginit + 'numb' + strgvarb, numb)
    setattr(gdat, strginit + 'indx' + strgvarb, indx)


def setp_cbar(gdat, strgcbar):

    minm = getattr(gdat, 'minmcntp' + strgcbar)
    maxm = getattr(gdat, 'maxmcntp' + strgcbar)
        
    retr_axis(gdat, strgcbar, minm, maxm, gdat.numbtickcbar - 1, scal=gdat.scalcntp)

    vmin = minm
    if gdat.scalcntp == 'asnh':
        vmin = np.arcsinh(vmin)
    vmax = maxm
    if gdat.scalcntp == 'asnh':
        vmax = np.arcsinh(vmax)

    tickscal = np.linspace(vmin, vmax, gdat.numbtickcbar)
    labl = np.empty(gdat.numbtickcbar, dtype=object)
    tick = np.copy(tickscal)
    for k in range(gdat.numbtickcbar):
        if gdat.scalcntp == 'asnh':
            tick[k] = np.sinh(tickscal[k])
        # avoid very small, but nonzero central values in the residual count color maps
        if strgcbar == 'cntpresi' and np.fabs(tick[k]) < 1e-5:
            tick[k] = 0.

        if strgcbar == 'cntpdata' and np.amax(tick) > 1e3:
            labl[k] = '%d' % tick[k]
        else:
            labl[k] = '%.3g' % tick[k]
    
    vmin = setattr(gdat, 'vmin' + strgcbar, vmin)
    vmax = setattr(gdat, 'vmax' + strgcbar, vmax)
    tick = setattr(gdat, 'tick' + strgcbar, tickscal)
    labl = setattr(gdat, 'labl' + strgcbar, labl)


def retr_cbar(gdat, strgcbar):
    
    vmin = getattr(gdat, 'vmin' + strgcbar)
    vmax = getattr(gdat, 'vmax' + strgcbar)
    tick = getattr(gdat, 'tick' + strgcbar)
    labl = getattr(gdat, 'labl' + strgcbar)

    return tick, labl, vmin, vmax


def retr_imagscal(gdat, imag):
    
    if gdat.scalcntp == 'linr':
        imagscal = imag
    else:
        imagscal = np.arcsinh(imag)

    return imagscal


def make_mock(gdat, gdatnotp):
    
    print 'Generating mock data...'
   
    if gdat.truefluxcent != None:
        boolcent = True
    else:
        boolcent = False
    
    gdat.truestdvlcpr = 1e-6
    gdat.truestdvcolr = np.array([0.05, 0.05])
    gdat.truemeancolr = np.array([0., 0.])
    gdat.truefluxdistslop = 2.
    
    if gdat.numbener > 3:
        raise Exception('')
    
    gdat.numbcolr = gdat.numbener - 1
    gdat.numblcpr = gdat.numbtime - 1
    
    indxstar = np.arange(gdat.truenumbstar)
    
    # position
    xpos = np.random.uniform(size=gdat.truenumbstar) * (gdat.sizeimag[0] - 1)
    ypos = np.random.uniform(size=gdat.truenumbstar) * (gdat.sizeimag[1] - 1)
    
    # flux
    flux = np.ones((gdat.numbener, gdat.numbtime, gdat.truenumbstar)) \
                * gdat.trueminmflux * np.exp(np.random.exponential(scale=1./(gdat.truefluxdistslop - 1.), size=gdat.truenumbstar))[None, None, :]
    
    if gdat.strgmodl == 'galx':
        xposgalx = np.random.uniform(size=gdat.truenumbgalx) * (gdat.sizeimag[0] - 1)
        yposgalx = np.random.uniform(size=gdat.truenumbgalx) * (gdat.sizeimag[1] - 1)
        fluxgalx = np.ones((gdat.numbener, gdat.numbtime, gdat.truenumbgalx)) \
                * gdat.trueminmfluxgalx * np.exp(np.random.exponential(scale=1./(gdat.truefluxdistslop - 1.), size=gdat.truenumbgalx))[None, None, :]

    print 'flux'
    summgene(flux)
    print 'fluxgalx'
    summgene(fluxgalx)

    if gdat.numbener > 1:
        # spectral parameters
        colr = gdat.stdvcolr[:gdat.numbener-1, None] * \
                        np.random.randn(gdat.numbcolr * gdat.truenumbstar).reshape((gdat.numbcolr, gdat.truenumbstar)) + \
                                                                                                                  gdat.meancolr[:gdat.numbener-1, None]
        for t in gdat.indxtime:
            flux[1:, t, :] *= 10**(0.4*colr)
    
    if gdat.numbtime > 1:
        # temporal parameters
        
        #arry = np.linspace(0., 1. - 1. / gdat.numblcpr, gdat.numblcpr)
        #temp = np.hstack(arry, gdat.numblcpr)
        #temp = (1e-6 * np.random.randn((gdat.numblcpr * gdat.truenumbstar)) + np.tile(np.linspace(0., 1. - 1. / gdat.numblcpr, gdat.numblcpr), (1, gdat.truenumbstar))) % 1.
        #temp = temp.reshape((gdat.numblcpr, gdat.truenumbstar)))
        temp = np.random.random((gdat.numblcpr * gdat.truenumbstar)).reshape((gdat.numblcpr, gdat.truenumbstar))
        temp = np.sort(temp, axis=0)
        temptemp = np.concatenate([np.zeros((1, gdat.truenumbstar))] + [temp] + [np.ones((1, gdat.truenumbstar))], axis=0)
        difftemp = temptemp[1:, :] - temptemp[:-1, :]
        flux[:, :, :] *= difftemp[None, :, :]
        
        # inject transits
        indxstartran = np.random.choice(indxstar, size=gdat.truenumbstar/2, replace=False)
        for k in indxstartran:
            indxinit = np.random.choice(gdat.indxtime)
            indxtemp = np.arange(indxinit, indxinit + 4) % gdat.numblcpr
            flux[:, indxtemp, k] *= np.random.rand()
        #flux[:, 1:, :] = fluxsumm[None, None, :] * lcpr[None, :, :]
    
    if boolcent:
        xpos[0] = np.array([0.5]) * (gdat.sizeimag[0] - 1)
        ypos[0] = np.array([0.5]) * (gdat.sizeimag[0] - 1)
        flux[:, :, 0] = gdat.truefluxcent
        
    if gdat.boolspre:
        xpos = xpos.astype(np.float32)
        ypos = ypos.astype(np.float32)
        flux = flux.astype(np.float32)
        if gdat.strgmodl == 'galx':
            xposgalx = xposgalx.astype(np.float32)
            yposgalx = yposgalx.astype(np.float32)
            fluxgalx = fluxgalx.astype(np.float32)
            
    gdat.catlrefr = [{}]
    gdat.catlrefr[0]['xpos'] = xpos
    gdat.catlrefr[0]['ypos'] = ypos
    gdat.catlrefr[0]['flux'] = flux
    if gdat.strgmodl == 'star':
        gdat.lablrefr = ['True']
        gdat.colrrefr = ['g']
    else:
        gdat.catlrefr.append({})
        gdat.lablrefr = ['True Star', 'True Galaxy']
        gdat.colrrefr = ['g', 'darkgreen']
        gdat.catlrefr[1]['xpos'] = xposgalx
        gdat.catlrefr[1]['ypos'] = yposgalx
        gdat.catlrefr[1]['flux'] = fluxgalx
    
    if gdat.verbtype > 1 and gdat.datatype == 'mock':
        print 'gdat.truenumbstar'
        print gdat.truenumbstar
        if gdat.strgmodl == 'galx':
            print 'gdat.truenumbgalx'
            print gdat.truenumbgalx

    if gdat.datatype == 'mock':
        gdat.cntpback = np.zeros((gdat.numbener, gdat.numbsidexpos, gdat.numbsideypos, gdat.numbtime)) + gdat.cntpbackscal
        print 'True background: %.g ADU' % gdat.cntpbackscal

    # evaluate model
    if gdat.strgmodl == 'star':
        xposeval = xpos
        yposeval = ypos
        fluxeval = flux
    else:
        paragalx = np.empty((gdat.numbparagalx, gdat.truenumbgalx))
        paragalx[gdat.indxxpos, :] = xposgalx
        paragalx[gdat.indxypos, :] = yposgalx
        paragalx[gdat.indxflux, :] = fluxgalx
        paragalx[[gdat.indxxxmo, gdat.indxxymo, gdat.indxyymo], :] = samp_prio_momegalx(gdat.trueminmradigalx, gdat.truenumbgalx)
        xposphon, yposphon, specphon = retr_tranphon(gdat.gridphon, gdat.amplphon, paragalx)
        xposeval = np.concatenate([xpos, xposphon])
        yposeval = np.concatenate([ypos, yposphon])
        fluxeval = np.concatenate([flux, specphon], axis=2)
    gdat.cntpdata = eval_modl(gdat, xposeval, yposeval, fluxeval, gdat.cntpback, clib=gdatnotp.clibeval, sizeimag=gdat.sizeimag)
    
    if not np.isfinite(gdat.cntpdata).all():
        print 'gdat.cntpdata'
        summgene(gdat.cntpdata)
        raise Exception('')

    gdat.cntpdata[gdat.cntpdata < 1] = 1.
    
    # add noise
    vari = gdat.cntpdata / gdat.gain
    gdat.cntpdata += (np.sqrt(vari) * np.random.normal(size=(gdat.numbener, gdat.sizeimag[1], gdat.sizeimag[0], gdat.numbtime))).astype(np.float32)
    
    gdat.cntpdata[gdat.cntpdata < 1.] = 1.

    arry = np.linspace(0.5, gdat.sizeimag[0] - 0.5, gdat.numbsidexpos)
    xposgrid, yposgrid = np.meshgrid(arry, arry)
    apho = np.empty((gdat.numbener, gdat.numbtime, gdat.truenumbstar))
    for k in range(len(xpos)):
        indxpixl = np.where(np.sqrt((xpos[k] - xposgrid)**2 + (ypos[k] - yposgrid)**2) < 3.)
        for i in gdat.indxener:
            for t in gdat.indxtime:
                apho[i, t, k] = np.sum(gdat.cntpdata[i, indxpixl[0], indxpixl[1], t])
    if not np.isfinite(gdat.cntpdata).all() or np.amin(gdat.cntpdata) < 0.:
        print 'gdat.cntpdata'
        summgene(gdat.cntpdata)
        raise Exception('')
    
    strg = ['%d' % k for k in range(gdat.truenumbstar)]
    # write to file
    path = gdat.pathdata + '%04d%04d%04d_mock.h5' % (gdat.numbener, gdat.numbtime, gdat.truenumbstar)
    print 'Writing to %s...' % path
    filetemp = h5py.File(path, 'w')
    filetemp.create_dataset('cntpdata', data=gdat.cntpdata)
    filetemp.create_dataset('numb', data=gdat.truenumbstar)
    filetemp.create_dataset('xpos', data=xpos)
    filetemp.create_dataset('strg', data=strg)
    filetemp.create_dataset('ypos', data=ypos)
    filetemp.create_dataset('flux', data=flux)
    filetemp.create_dataset('gain', data=gdat.gain)
    filetemp.create_dataset('apho', data=apho)
    filetemp.close()
    
    gdat.truecatl = read_catl(gdat, path, boolmock=True)
    if gdat.inittype == 'rand':
        gdat.catlinit = None
    else:
        gdat.catlinit = gdat.truecatl
    
    
def to_moments(r, theta, phi):
    u = np.cos(theta)
    xx = r*r*(1.+u*u+(1.-u*u)*np.cos(2.*phi))/4.
    xy = r*r*(1.-u*u)*np.sin(2.*phi)/4.
    yy = r*r*(1.+u*u-(1.-u*u)*np.cos(2.*phi))/4.
    if (xy*xy > xx*yy).any():
     mask = xy*xy > xx*yy
     print r[mask], theta[mask]/np.pi, phi[mask]/np.pi
     print np.cos(theta[mask])
     print xx[mask], xy[mask], yy[mask]
     print xx[mask]*yy[mask], xy[mask]*xy[mask]
     assert False
    return xx, xy, yy

def from_moments(xx, xy, yy):
    assert (xx*yy > xy*xy).all()
    a = np.sqrt(xx+yy+np.sqrt((xx-yy)*(xx-yy)+4*xy*xy))
    b = np.sqrt(xx+yy-np.sqrt((xx-yy)*(xx-yy)+4*xy*xy))
    theta = np.arccos(b/a)
    phi = np.arctan2(2*xy, -(yy-xx))/2.
    return a, theta, phi 


def retr_factsers(sersindx):
    
    factsers = 1.9992 * sersindx - 0.3271
    
    return factsers


def retr_sers(gdat, numbsidegrid=21, sersindx=4., factusam=101, factradigalx=2., pathplot=None):
    
    # numbsidegrid is the number of pixels along one side of the grid and should be an odd number
    numbsidegridusam = numbsidegrid * factusam

    # generate the grid
    xpostemp = np.linspace(-factradigalx, factradigalx, numbsidegridusam) #+1
    ypostemp = np.linspace(-factradigalx, factradigalx, numbsidegridusam) #+1
    xpos, ypos = np.meshgrid(xpostemp, ypostemp)

    # evaluate the Sersic profile
    ## approximation to the b_n factor
    factsers = retr_factsers(sersindx)
    ## profile sampled over the square grid
    amplphonusam = np.exp(-factsers * (np.sqrt(xpos**2 + ypos**2)**(1. / sersindx) - 1.))#.flatten()
    ## down sample
    amplphon = np.empty((numbsidegrid, numbsidegrid)) #+1
    for k in range(numbsidegrid): #+1
        for l in range(numbsidegrid): #+1
            amplphon[k, l] = np.mean(amplphonusam[k*factusam:(k+1)*factusam, l*factusam:(l+1)*factusam])
    indx = np.linspace(factusam / 2, numbsidegridusam - factusam / 2 + 1, numbsidegrid, dtype=int)

    xpostemp = xpostemp[indx]
    ypostemp = ypostemp[indx]

    xpos, ypos = np.meshgrid(xpostemp, ypostemp)

    amplphon = amplphon.flatten()
    ## normalize such that the sum is unity
    amplphon /= sum(amplphon)

    gridphon = np.vstack((xpos.flatten(), ypos.flatten()))

    return gridphon, amplphon


def retr_tranphon(gridphon, amplphon, galaxypars, convert=True):
    
    xposgalx, yposgalx, fluxgalx, arg1, arg2, arg3 = galaxypars
    if convert:
        radigalx, thetagalx, phigalx = from_moments(arg1, arg2, arg3)
        radigalx *= 1.33846609401 / np.sqrt(2) # only works for Sersic index 2, makes <x^2>=1
    else:
        radigalx, thetagalx, phigalx = arg1, arg2, arg3
    tranmatr = np.empty((xposgalx.size, 2, 2))
    tranmatr[:,0,0] = radigalx * np.cos(phigalx)
    tranmatr[:,0,1] = -radigalx * np.cos(thetagalx) * np.sin(phigalx)
    tranmatr[:,1,0] = radigalx * np.sin(phigalx)
    tranmatr[:,1,1] = radigalx * np.cos(thetagalx) * np.cos(phigalx)

    # transform the phonion grid
    gridphontran = np.matmul(tranmatr, gridphon)

    # positions of the phonions
    xposphon = xposgalx[:, None] + gridphontran[:, 0, :]
    yposphon = yposgalx[:, None] + gridphontran[:, 1, :]

    # spectrum of the phonions
    specphon = fluxgalx[:, None] * amplphon[None, :]
    
    specphon = specphon.flatten()[None, None, :]

    return (xposphon.flatten()).astype(np.float32), (yposphon.flatten()).astype(np.float32), specphon.astype(np.float32)


def plot_fluxhist(gdat, flux, i, t, jj=None, plotmagt=False):
    
    figr, axis = plt.subplots()
    gdat.deltfluxplot = gdat.binsfluxplot[1:] - gdat.binsfluxplot[:-1]
    if plotmagt:
        xdat = gdat.binsmagtplot[:-1] + gdat.deltmagtplot / 2.
        delt = gdat.deltmagtplot
    else:
        xdat = gdat.binsfluxplot[:-1] + gdat.deltfluxplot / 2.
        delt = gdat.deltfluxplot
    for k in gdat.indxrefr:
        if plotmagt:
            magtrefr = -2.5 * np.log10(gdat.catlrefr[k]['flux'][i, t, :]) + gdat.magtzero
            hist = np.histogram(magtrefr, bins=gdat.binsfluxplot)[0]
        else:
            hist = np.histogram(gdat.catlrefr[k]['flux'][i, t, :], bins=gdat.binsfluxplot)[0]
        axis.bar(xdat, hist, delt, alpha=0.5, label=gdat.lablrefr[k], lw=gdat.linewdth, facecolor=gdat.colrrefr[k], edgecolor=gdat.colrrefr[k])
    # temp
    if plotmagt:
        magt = -2.5 * np.log10(flux[i, t, :]) + gdat.magtzero
        hist = np.histogram(magt, bins=gdat.binsmagtplot)[0]
    else:
        hist = np.histogram(flux[i, t, :], bins=gdat.binsfluxplot)[0]
    if jj != None:
        colr = 'b'
        labl = 'Condensed'
    else:
        colr = 'y'
        labl = 'Sample'
    axis.bar(xdat, hist, delt, edgecolor=colr, facecolor=colr, lw=gdat.linewdth, alpha=0.5, label=labl)
    if plotmagt:
        axis.set_xlim([gdat.minmmagtplot, gdat.maxmmagtplot])
    else:
        axis.set_xlim([gdat.minmfluxplot, gdat.maxmfluxplot])
    axis.set_ylim(gdat.limthist)
    axis.set_xscale('log')
    axis.set_yscale('log')
    plt.tight_layout()
    if plotmagt:
        strg = 'magt'
    else:
        strg = 'flux'
    if jj == None:
        plt.savefig(gdat.pathdatartag + '%s_hist%s%04d%04d.%s' % (gdat.rtag, strg, i, t, gdat.strgplotfile))
    else:
        plt.savefig(gdat.pathdatartag + '%s_hist%s%04d%04d_fram%04d.%s' % (gdat.rtag, strg, i, t, jj, gdat.strgplotfile))
    plt.close()


def plot_imag(gdat, cntp, strgcbar, strgfile):
    
    for i in gdat.indxener:
        for t in gdat.indxtime:
            figr, axis = plt.subplots()
            tick, labl, vmin, vmax = retr_cbar(gdat, strgcbar)
            imagscal = retr_imagscal(gdat, cntp)
            imag = axis.imshow(imagscal[i, :, :, t], origin='lower', interpolation='nearest', cmap='Greys_r', vmin=vmin, vmax=vmax)
            setp_imaglimt(gdat, axis)
            cbar = plt.colorbar(imag, ax=axis, fraction=0.05, aspect=15)
            cbar.set_ticks(tick)
            cbar.set_ticklabels(labl)
            supr_catl(gdat, axis, i, t)
            plt.tight_layout()
            plt.savefig(gdat.pathdatartag + '%s_%s_%02d%02d.%s' % (gdat.rtag, strgfile, i, t, gdat.strgplotfile))
            plt.close()


def retr_spec(gdat, flux, colr):
    
    if gdat.colrtype == 'powr':
        spec = flux * (gdat.meanener / gdat.enerpivt)**(-colr)
    
    return spec


def main( \
         # string characterizing the type of data
         strgdata='sdss0921', \
        
         # numpy array containing the data counts
         cntpdata=None, \
        
         # weight
         weig=None, \

         # scalar bias
         bias=None, \
         
         # string characterizing the type of PSF
         strgpsfn='sdss0921', \
        
         # numpy array containing the PSF counts
         cntppsfnusam=None, \
            
         # factor by which the PSF is oversampled
         factusam=None, \
         
         # nonnormalized probabilities of proposals
         probprop=None, \
        
         # margin size
         marg=10, \

         inittype='refr', \

         initxpos=None, \
         initypox=None, \
        
         boolplotinit=True, \

         # data path
         pathdatartag=None, \

         # a string defining the experiment type
         exprtype=None, \
            
         boolframplotenti=False, \

         boolbili=False, \

         # Boolean flag to turn on single-precision
         boolspre=True, \
            
         labldata=None, \
         labldataaxis=None, \

         truefluxcent=None, \

         # run tag to be post-processed
         strgproc=None, \
        
         fudgpena=1., \

         # Boolean flag to control tiling
         booltile=True, \
        
         # Boolean flag to control C library usage
         boolclib=True, \
            
         priotype='prio', \

         # configuration string
         strgcnfg=None, \

         # level of background 
         cntpback=None, \

         # a string indicating the name of the state
         strgstat=None, \
         boolstatread=True, \

         # user-defined time stamp string
         strgtimestmp=None, \

         # number of samples
         numbswep=100, \
    
         # size of the regions in number of pixels
         # temp
         sizeregi=None, \
        
         magtzero=0., \

         # string indicating type of plot file
         strgplotfile='pdf', \

         # number of samples
         numbswepburn=None, \
    
         # factor by which the sweeps will be thinned
         factthin=None, \
    
         # factor by which the framw plots will be thinned
         numbplotfram=20, \
    
         # catalog to initialize the chain with
         catlinit=None, \

         # number of loops
         numbloop=1000, \
         
         # string indicating whether the data is simulated or input by the user
         # 'mock' for mock data, 'inpt' for input data
         datatype='mock', \

         numbsidexpos=100, \
         numbsideypos=100, \

         # boolean flag whether to save the image and catalog samples to disc
         boolplotsave=True, \
       
         # arbitrary normalization of the standard deviation of the position proposals
         stdvposiprop=None, \

         # arbitrary normalization of the standard deviation of the flux proposals
         stdvfluxprop=None, \

         # string indicating the type of model
         strgmode='pcat', \
        
         numbener=None, \
         numbtime=None, \
         
         # reference catalog
         catlrefr=None, \

         # labels for reference catalogs
         lablrefr=None, \

         # colors for reference catalogs
         colrrefr=None, \

         # a string extension to the run tag
         rtagextn=None, \
       
         # level of verbosity
         verbtype=1, \
         
         # diagnostic mode
         diagmode=False, \
        
         # color style
         colrstyl='pcat', \

         # string to indicate the ingredients of the photometric model
         # 'star' for star only, 'galx' for star and galaxy
         strgmodl='star', \

         **args \

         ):
    
    np.random.seed(0)

    # construct the global object 
    gdat = gdatstrt()
    ## not picklable
    
    # copy all provided inputs to the global object
    for strg, valu in args.iteritems():
        setattr(gdat, strg, valu)

    # load arguments into the global object
    for attr, valu in locals().iteritems():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)
    
    gdatnotp = gdatstrt()

    np.seterr(all='warn')
    
    ## name of the configuration function
    if gdat.strgcnfg is None:
        gdat.strgcnfg = inspect.stack()[1][3]

    gdat.pathlion, gdat.pathdata = retr_path()
    
    if gdat.cntpdata is None:
        gdat.datatype = 'mock'
    else:
        gdat.datatype = 'inpt'
    
    if gdat.datatype == 'mock':
        if gdat.numbener is None:
            gdat.numbener = 1
        if gdat.numbtime is None:
            gdat.numbtime = 1

    if gdat.numbener > 1:
        gdat.indxenerpivt = gdat.numbener / 2
        gdat.enerpivt = gdat.meanener[gdat.indxenerpivt]

    if gdat.strgproc is None:
        
        if gdat.cntpdata is None:
            if gdat.numbsidexpos is None:
                gdat.numbsidexpos = 40
            if gdat.numbsideypos is None:
                gdat.numbsideypos = 40
        else:
            gdat.numbsidexpos = gdat.cntpdata.shape[2]
            gdat.numbsideypos = gdat.cntpdata.shape[1]

        # setup
        setp(gdat)
        
        # diagnostic variables
        gdat.strgvarbdiag = ['Acceptance', 'Out of Bounds', 'Proposal (ms)', 'Likelihood (ms)', 'Implement (ms)']
        gdat.numbvarbdiag = len(gdat.strgvarbdiag)
        gdat.indxvarbdiag = np.arange(gdat.numbvarbdiag)

        setp_varbvalu(gdat, 'fluxdistslop', 2.)
        setp_varbvalu(gdat, 'fluxdistslopgalx', 2.)
        setp_varbvalu(gdat, 'minmflux', 1000.)
        setp_varbvalu(gdat, 'minmradigalx', 1.)
        setp_varbvalu(gdat, 'minmfluxgalx', 2000.)
        setp_varbvalu(gdat, 'cntpbackscal', 100., comm=True)
        setp_varbvalu(gdat, 'maxmnumbstar', 2000, comm=True)
        setp_varbvalu(gdat, 'maxmnumbgalx', 2000, comm=True)
        setp_varbvalu(gdat, 'numbgalx', 100, strgmodl='true')
        setp_varbvalu(gdat, 'numbstar', 10, strgmodl='true')
        setp_varbvalu(gdat, 'gain', 1., comm=True)
        setp_varbvalu(gdat, 'cntpback', np.full(gdat.sizeimag, 225.), comm=True)

        if gdat.datatype == 'inpt':
            gdat.numbtime = gdat.cntpdata.shape[3]
            gdat.numbener = gdat.cntpdata.shape[0]
        
        # defaults
        ## time stamp string
        if gdat.strgtimestmp is None:
            gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  
        ## number of sweeps to be burnt
        if gdat.numbswepburn is None:
            gdat.numbswepburn = int(0.2 * gdat.numbswep)
        gdat.numbswlp = gdat.numbswep * gdat.numbloop
        gdat.numbswlpburn = gdat.numbswepburn * gdat.numbloop

        gdat.numbsamp = gdat.numbswep - gdat.numbswepburn
    
        # run tag
        gdat.rtag = 'pcat_' + gdat.strgtimestmp + '_' + gdat.strgcnfg + '_%06d' % gdat.numbswlp
        
        if gdat.rtagextn != None:
            gdat.rtag += '_' + gdat.rtagextn
    
        gdat.indxsamp = np.arange(gdat.numbsamp)
        
        if gdat.cntppsfnusam is None:
            retr_cntppsfnusam(gdat)
        else:
            gdat.numbsidepsfnusam = gdat.cntppsfnusam.shape[1]
            gdat.numbsidepsfn = gdat.numbsidepsfnusam / gdat.factusam
        
        gdat.numbpixlpsfn = gdat.numbsidepsfn**2
        gdat.boolsaveswep = np.zeros(gdat.numbswep, dtype=bool)
        gdat.indxswepsave = np.arange(gdat.numbswepburn, gdat.numbswepburn + gdat.numbsamp)
        gdat.boolsaveswep[gdat.indxswepsave] = True

        print 'Lion initialized at %s' % gdat.strgtimestmp

        # show critical inputs
        print 'strgdata: ', gdat.strgdata
        print 'Model type:', gdat.strgmodl
        print 'Data type:', datatype
        print 'Photometry mode: ', strgmode
        print 'booltile: ', gdat.booltile
        print 'sizeregi: ', gdat.sizeregi
        print 'boolspre: ', gdat.boolspre
        print 'gdat.numbswep: ', gdat.numbswep
        print 'gdat.numbsamp: ', gdat.numbsamp
        print 'gdat.numbloop: ', gdat.numbloop
        print 'gdat.gain: ', gdat.gain
        print 'gdat.fittminmflux', gdat.fittminmflux

        gdat.cmapresi = make_cmapdivg('Red', 'Orange')
        if colrstyl == 'pcat':
            gdat.sizemrkr = 3.
            gdat.linewdth = 3
            gdat.colrbrgt = 'green'
        
        gdat.indxtime = np.arange(gdat.numbtime)
        gdat.indxener = np.arange(gdat.numbener)
        
        if gdat.pathdatartag is None:
            gdat.pathdatartag = gdat.pathdata + gdat.rtag + '/'
        
        os.system('mkdir -p %s' % gdat.pathdatartag)

        retr_coefspix(gdat)
        
        if gdat.strgmodl == 'galx':
            gdat.gridphon, gdat.amplphon = retr_sers(gdat)
  
        # construct C library
        setp_clib(gdat, gdatnotp, gdat.pathlion)

        if gdat.boolclib:
            gdatnotp.clibeval = gdatnotp.clib.clib_eval_modl
        else:
            gdatnotp.clibeval = None

        cntr = cntrstrt()
        gdat.indxxpos = cntr.incr()
        gdat.indxypos = cntr.incr()
        gdat.indxflux = np.zeros((gdat.numbener, gdat.numbtime), dtype=int)
        for k in gdat.indxener:
            for l in gdat.indxtime:
                gdat.indxflux[k, l] = cntr.incr()
        gdat.numbparastar = cntr.gets()
        if gdat.strgmodl == 'galx':
            gdat.indxxxmo = cntr.incr()
            gdat.indxxymo = cntr.incr()
            gdat.indxyymo = cntr.incr()
        gdat.numbparagalx = cntr.gets()
        gdat.indxprvl = cntr.incr()
        gdat.numbparacatlcond = 1 + gdat.numbparastar
        if gdat.strgmodl == 'galx':
            gdat.numbparacatlcond = 1 + gdat.numbparagalx
        
        # generate mock data
        if gdat.cntpdata is None:
            make_mock(gdat, gdatnotp)

        if gdat.datatype == 'inpt':
            gdat.numbsideypos = gdat.cntpdata.shape[1]
            gdat.numbsidexpos = gdat.cntpdata.shape[2]
    
        print 'gdat.numbener: ', gdat.numbener
        print 'gdat.numbtime: ', gdat.numbtime
        
        if gdat.cntpdata.ndim != 4:
            raise Exception('Input image should be four dimensional.')
        
        gdat.vari = gdat.cntpdata / gdat.gain
        if gdat.weig is None:
            gdat.weig = 1. / gdat.vari # inverse variance
        print 'gdat.weig'
        summgene(gdat.weig)
        
        if not np.isfinite(gdat.cntpdata).all() or np.amin(gdat.cntpdata) < 0.:
            print 'Input count map goes either negative or infinite.'
            print 'gdat.cntpdata'
            summgene(gdat.cntpdata)
            #raise Exception('')

        if not np.isfinite(gdat.vari).all() or np.amin(gdat.vari) < 0.:
            print 'Variance map goes either negative or infinite.'
            print 'gdat.vari'
            summgene(gdat.vari)
            #raise Exception('')

        if not np.isfinite(gdat.weig).all() or np.amin(gdat.weig) < 0.:
            print 'Weight map goes either negative or infinite.'
            print 'gdat.weig'
            summgene(gdat.weig)
            #raise Exception('')
        
        if gdat.boolspre:
            if not isinstance(gdat.cntpdata, np.float32):
                gdat.cntpdata = gdat.cntpdata.astype(np.float32)
                gdat.weig = gdat.weig.astype(np.float32)
        
        assert gdat.cntpdata.shape[0] == gdat.cntppsfnusam.shape[0]

        if gdat.boolstatread and gdat.strgstat != None and gdat.catlinit != None:
            print 'strgstat and catlinit defined at the same time. catlinit takes precedence for the initial catalog.'

        # plotting
    
        gdat.factthinplot = gdat.numbswep / gdat.numbplotfram

        gdat.maxmcntpresi = 2. * np.sqrt(np.median(gdat.cntpdata))
        gdat.minmcntpresi = -gdat.maxmcntpresi
        gdat.minmcntpdata = np.percentile(gdat.cntpdata, 5)
        gdat.maxmcntpdata = np.percentile(gdat.cntpdata, 95)
        gdat.limthist = [0.5, 1e3]
        gdat.numbbinsfluxplot = 10
        gdat.minmfluxplot = 1e1
        gdat.maxmfluxplot = 1e6
        gdat.binsfluxplot = 10**(np.linspace(np.log10(gdat.minmfluxplot), np.log10(gdat.maxmfluxplot), gdat.numbbinsfluxplot + 1))
        gdat.binsmagtplot = -2.5 * np.log10(gdat.binsfluxplot[::-1]) + gdat.magtzero
        gdat.deltmagtplot = gdat.binsmagtplot[1:] - gdat.binsmagtplot[:-1]
        gdat.meanmagtplot = (gdat.binsmagtplot[1:] + gdat.binsmagtplot[:-1]) / 2.
        gdat.minmmagtplot = 20.
        gdat.maxmmagtplot = 3.
        gdat.numbtickcbar = 11

        gdat.scalcntp = 'asnh'
        setp_cbar(gdat, 'data')
        setp_cbar(gdat, 'resi')
        
        if gdat.datatype != 'mock':
            if gdat.catlrefr is None:
                gdat.catlrefr = []

        print 'len(gdat.catlrefr)'
        print len(gdat.catlrefr)
        gdat.numbrefr = len(gdat.catlrefr)
        gdat.indxrefr = np.arange(gdat.numbrefr)
        
        if gdat.probprop is None:
            if strgmode == 'pcat':
                
                gdat.probprop = np.array([80., 40., 40.])
                    
                if gdat.strgmodl == 'galx':
                    gdat.probprop = np.array([80., 40., 40., 80., 40., 40., 0., 40., 0.])
            else:
                gdat.probprop = np.array([80., 0., 0.])
        gdat.probprop /= np.sum(gdat.probprop)
        
        if gdat.stdvposiprop is None:
            gdat.stdvposiprop = 12.
        
        if gdat.stdvfluxprop is None:
            gdat.stdvfluxprop = 1.
        
        if gdat.booltile:
            gdat.marg = 10
        else:
            gdat.marg = 0
            assert gdat.sizeregi == gdat.sizeimag[0]
            assert gdat.sizeregi == gdat.sizeimag[1]

        gdat.listproptype = ['P *', 'BD *', 'MS *', 'P g', 'BD g', '*-g', '**-g', '*g-g', 'MS g']
    
        if gdat.strgmode != 'pcat' and gdat.catlinit == None:
            raise Exception('')

        if gdat.strgmode != 'pcat':
            gdat.catlinit = gdat.catlrefr[0]
        
        gdat.boolinitcatl = False
        if gdat.strgstat != None:
            gdat.pathstat = gdat.pathdata + 'stat_' + gdat.strgstat + '.h5'
            if gdat.boolstatread: 
                if os.path.isfile(gdat.pathstat):
                    gdat.catlinit = read_catl(gdat, gdat.pathstat)
                    gdat.boolinitcatl = True
                else:
                    print 'No initialization catalog found. Initializing randomly...'

        # check if source has been changed after compilation
        if os.path.getmtime(gdat.pathlion + 'blas.c') > os.path.getmtime(gdat.pathlion + 'blas.so'):
            raise Exception('blas.c modified after compiled blas.so')
            #warnings.warn('blas.c modified after compiled blas.so', Warning)
        
        # constants
        ## number of bands (i.e., energy bins)
        
        gdat.boolplot = gdat.boolplotsave

        # parse PSF
        ## number of pixels along the side of the upsampled PSF
        #gdat.numbsidepsfnusam = cntppsfn.shape[1]
        #numbsidepsfn = gdat.numbsidepsfnusam / factusam
        
        if gdat.verbtype > 2:
            print 'gdat.numbsidepsfn'
            print gdat.numbsidepsfn
            print 'factusam'
            print factusam
  
        # read data
        if not isinstance(gdat.cntpdata, np.ndarray):
            raise Exception('')
        

        numbpixl = gdat.numbsidexpos * gdat.numbsideypos

        gdat.numbflux = gdat.numbener * gdat.numbtime

        print 'Image width and height: %d %d pixels' % (gdat.numbsidexpos, gdat.numbsideypos)
        
        gdat.numbdata = numbpixl * gdat.numbtime
  
        # hyperparameters
        gdat.stdvcolr = [0.5, 0.5]
        gdat.meancolr = [0.25, 0.1]
        gdat.stdvlcpr = 1e-4

        # plots
        if gdat.boolplot and gdat.boolplotinit:
            
            # background
            print 'gdat.cntpback'
            summgene(gdat.cntpback)
            print 'gdat.cntpdata'
            summgene(gdat.cntpdata)
            plot_imag(gdat, gdat.cntpback, 'data', 'cntpback')

            ## PSF
            arry = [0., 0.25, 0.4, 0.45, 0.49, 0.5, 0.51, 0.55, 0.6, 0.75, 1.]
            for k in range(len(arry)):
                plot_psfn(gdat, k, arry[k], arry[k], clib=gdatnotp.clibeval)
            indx = len(arry)
            plot_psfn(gdat, indx, -0.5, 0.5, clib=gdatnotp.clibeval)
            plot_psfn(gdat, indx + 1, 0.5, -0.5, clib=gdatnotp.clibeval)
            plot_psfn(gdat, indx + 2, -0.5, -0.5, clib=gdatnotp.clibeval)
            plot_psfn(gdat, indx + 3, -1, -1, clib=gdatnotp.clibeval)
            plot_psfn(gdat, indx + 4, 1, -1, clib=gdatnotp.clibeval)
            plot_psfn(gdat, indx + 5, -1, 1, clib=gdatnotp.clibeval)
            plot_psfn(gdat, indx + 6, 1, 1, clib=gdatnotp.clibeval)

            ## data
            if gdat.numbener > 1 or gdat.numbtime > 1:
                for i in gdat.indxener:
                    for t in gdat.indxtime:

                        
                        figr, axis = plt.subplots()
                        tick, labl, vmin, vmax = retr_cbar(gdat, 'data')
                        imagscal = retr_imagscal(gdat, gdat.cntpdata[i, :, :, t])
                        imag = axis.imshow(imagscal, origin='lower', interpolation='nearest', cmap='Greys_r', vmin=vmin, vmax=vmax)
                        ## limits
                        setp_imaglimt(gdat, axis)
                        cbar = plt.colorbar(imag, ax=axis, fraction=0.05, aspect=15)
                        cbar.set_ticks(tick)
                        cbar.set_ticklabels(labl)
                        supr_catl(gdat, axis, i, t)
                        plt.tight_layout()
                        plt.savefig(gdat.pathdatartag + '%s_cntpdataenti_fram%04d%04d.%s' % (gdat.rtag, i, t, gdat.strgplotfile))
                        
                        plt.close()
                print 'Making animations of cadence images...'
                cmnd = 'convert -delay 20 -density 200x200 %s/%s_cntpdataenti_fram*.%s %s/%s_cntpdataenti.gif' % \
                                                                            (gdat.pathdatartag, gdat.rtag, gdat.strgplotfile, gdat.pathdatartag, gdat.rtag)
                print cmnd
                os.system(cmnd)
                print 'Done.'
        
        
        class Proposal:
        
            def __init__(self):
                
                if gdat.verbtype > 1:
                    print 'Initializing the proposal...'

                self.idx_move = None
                self.idx_move_g = None
                self.do_birth = False
                self.do_birth_g = False
                self.idx_kill = None
                self.idx_kill_g = None
                self.factor = None
                self.goodmove = False
                
                if gdat.boolspre:
                    self.xphon = np.array([], dtype=np.float32)
                    self.yphon = np.array([], dtype=np.float32)
                    self.fphon = np.array([[[]]], dtype=np.float32)
                else:
                    self.xphon = np.array([])
                    self.yphon = np.array([])
                    self.fphon = np.array([[[]]])

            
            def set_factor(self, factor):
                self.factor = factor
        
            
            def assert_types(self):
                assert self.xphon.dtype == np.float32
                assert self.yphon.dtype == np.float32
                assert self.fphon.dtype == np.float32
        
            
            def __add_phonions_stars(self, stars, remove=False):
                fluxmult = -1 if remove else 1
                
                if gdat.verbtype > 1:
                    print '__add_phonions_stars()'
                
                if gdat.verbtype > 2:
                    print 'before:'
                    print 'self.xphon'
                    summgene(self.xphon)
                    print 'stars'
                    print stars
                    summgene(stars)
                    print 'stars[gdat.indxxpos, :]'
                    summgene(stars[gdat.indxxpos, :])
                
                self.xphon = np.append(self.xphon, stars[gdat.indxxpos, :])
                
                if gdat.verbtype > 2:
                    print 'after:'
                    print 'self.xphon'
                    summgene(self.xphon)
                
                self.yphon = np.append(self.yphon, stars[gdat.indxypos, :])
                if self.fphon.size == 0:
                    self.fphon = np.copy(fluxmult*stars[gdat.indxflux, :])
                else:
                    self.fphon = np.append(self.fphon, fluxmult*stars[gdat.indxflux, :], axis=2)
                
                if gdat.boolspre:
                    self.assert_types()
        
            
            def __add_phonions_galaxies(self, galaxies, remove=False):
                fluxmult = -1 if remove else 1
                xtemp, ytemp, ftemp = retr_tranphon(gdat.gridphon, gdat.amplphon, galaxies)
                self.xphon = np.append(self.xphon, xtemp)
                self.yphon = np.append(self.yphon, ytemp)
                self.fphon = np.append(self.fphon, fluxmult*ftemp, axis=2)
                if gdat.boolspre:
                    self.assert_types()
        
            
            def add_move_stars(self, idx_move, stars0, starsp):
                
                if gdat.verbtype > 2:
                    print 'add_move_stars()'

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
                    #print 'self.stars0[gdat.indxxpos,:]'
                    #summgene(self.stars0[gdat.indxxpos,:])
                    #print 'self.stars0[gdat.indxypos,:]'
                    #summgene(self.stars0[gdat.indxypos,:])
                    #print
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

        class Model:
            
            # should these be class or instance variables?
        
            trueminf_g = np.float32(250.)
            truealpha_g = np.float32(2.00)
            gdat.trueminmradigalx = np.float32(1.00)
            
            if gdat.boolspre and gdat.datatype == 'mock':
                #trueminf = np.float32(trueminf)
                gdat.truefluxdistslop = np.float32(gdat.truefluxdistslop)
            
            # temp
            penalty = gdat.fudgpena * 0.5 * gdat.numbparastar
            penalty_g = 3.0
            scalspmrposi = 1.
            scalspmrposi_g = 1.
            
            def __init__(self):

                # initialize parameters
                if not gdat.boolinitcatl:
                    self.n = np.random.randint(gdat.maxmnumbstar) + 1
                else:
                    self.n = gdat.catlinit['numb']
                if gdat.verbtype > 1:
                    print 'Initializing the model...'
                if gdat.boolspre:
                    self.stars = np.zeros((gdat.numbparastar, gdat.maxmnumbstar), dtype=np.float32)
                else:
                    self.stars = np.zeros((gdat.numbparastar, gdat.maxmnumbstar), dtype=np.float32)
                
                if gdat.verbtype > 2:
                    print 'self.stars'
                    summgene(self.stars)
                
                if gdat.initxpos is not None:
                    if gdat.initxpos < 0. or gdat.initxpos > gdat.sizeimag[0] or gdat.initxpos > gdat.sizeimag[1]:
                        raise Exception('')
                    if gdat.initypos < 0. or gdat.initypos > gdat.sizeimag[0] or gdat.initypos > gdat.sizeimag[1]:
                        raise Exception('')
                    self.stars[:, :self.n] = gdat.initxpos
                    self.stars[:, :self.n] = gdat.initypos
                elif gdat.catlinit is None and gdat.strgmode == 'pcat':
                    self.stars[:, :self.n] = np.random.uniform(size=(gdat.numbparastar, self.n))  # refactor into some sort of prior function?
                    self.stars[gdat.indxxpos, :self.n] *= gdat.sizeimag[0] - 1
                    self.stars[gdat.indxypos, :self.n] *= gdat.sizeimag[1] - 1
                else:
                    print 'Initializing positions from the catalog...'
                    self.stars[gdat.indxxpos, :gdat.catlinit['numb']] = gdat.catlinit['xpos'][:gdat.catlinit['numb']]
                    self.stars[gdat.indxypos, :gdat.catlinit['numb']] = gdat.catlinit['ypos'][:gdat.catlinit['numb']]
                    
                    indxbadd = np.where((self.stars[gdat.indxxpos, :] < 0.) | (self.stars[gdat.indxypos, :] < 0.) | \
                                        (self.stars[gdat.indxxpos, :] > gdat.numbsidexpos) | (self.stars[gdat.indxypos, :] > gdat.numbsideypos))
                    if indxbadd[0].size > 0:
                        print 'Some of the initialization positions are outside the positional priors.'
                        raise Exception('')
                
                if gdat.catlinit is None:
                    print 'Initializing the fluxes randomly...'
                    self.stars[gdat.indxflux, :self.n] = np.random.uniform(size=(gdat.numbflux, self.n))  # refactor into some sort of prior function?
                    self.stars[gdat.indxflux, :self.n] **= -1. / (gdat.fittfluxdistslop - 1.)
                    self.stars[gdat.indxflux, :self.n] *= gdat.fittminmflux
                else:
                    print 'Initializing fluxes from the catalog...'
                    #print 'gdat.indxflux'
                    #summgene(gdat.indxflux)
                    #print 'self.stars'
                    #summgene(self.stars)
                    #print 'self.stars[gdat.indxflux, :gdat.catlinit[numb]'
                    #summgene(self.stars[gdat.indxflux, :gdat.catlinit['numb']])
                    numbtemp = min(gdat.catlinit['numb'], gdat.maxmnumbstar)
                    self.stars[gdat.indxflux, :numbtemp] = gdat.catlinit['flux'][:, :, :numbtemp]
            
                    for i in gdat.indxener:
                        for t in gdat.indxtime:
                            indxbadd = np.where((self.stars[gdat.indxflux[i, t], :] < gdat.fittminmflux) & (self.stars[gdat.indxflux[i, t], :] > 0.))[0]
                            if indxbadd.size > 0:
                                print 'self.stars[gdat.indxflux[i, t], :]'
                                print self.stars[gdat.indxflux[i, t], :]
                                print 'gdat.fittminmflux'
                                print gdat.fittminmflux
                                print 'Some of the initialization fluxes were below the minimum flux prior. Initializing them at the minimum flux.'
                                raise Exception('')
                                self.stars[gdat.indxflux[i, t], indxbadd] = np.random.uniform(size=indxbadd.size)
                                self.stars[gdat.indxflux[i, t], indxbadd] **= -1. / (gdat.fittfluxdistslop - 1.)
                                self.stars[gdat.indxflux[i, t], indxbadd] *= gdat.fittminmflux
                
                #for k in range(len(gdat.catlinit['xpos'])):
                #    print 'gdat.catlinit'
                #    print gdat.catlinit['xpos'][k], gdat.catlinit['ypos'][k], gdat.catlinit['flux'][:, :, k].flatten()
                #    print 'self.stars'
                #    print self.stars[gdat.indxxpos, k], self.stars[gdat.indxypos, k], self.stars[gdat.indxflux.flatten(), k]
                #    print

                if gdat.verbtype > 2:
                    print 'numb'
                    print self.n
                    for strgfeat in gdat.liststrgfeatstar:
                        indx = getattr(gdat, 'indx' + strgfeat)
                        print strgfeat
                        summgene(self.stars[indx, :self.n])
                        print

                self.ng = 0
                if gdat.strgmodl == 'galx':
                    self.ng = np.random.randint(gdat.maxmnumbgalx)+1
                    self.galaxies = np.zeros((6,gdat.maxmnumbgalx), dtype=np.float32)
                    # temp -- 3 should be generalized to temporal modeling
                    self.galaxies[[gdat.indxxpos,gdat.indxypos,gdat.indxflux],0:self.ng] = np.random.uniform(size=(3, self.ng))
                    self.galaxies[gdat.indxxpos, :self.ng] *= gdat.sizeimag[0] - 1
                    self.galaxies[gdat.indxypos, :self.ng] *= gdat.sizeimag[1]-1
                    self.galaxies[gdat.indxflux, :self.ng] **= -1./(gdat.fittfluxdistslopgalx - 1.)
                    self.galaxies[gdat.indxflux, :self.ng] *= gdat.fittminmfluxgalx
                    print 'self.galaxies[gdat.indxflux, :self.ng]'
                    summgene(self.galaxies[gdat.indxflux, :self.ng])
                    self.galaxies[[gdat.indxxxmo,gdat.indxxymo,gdat.indxyymo], :self.ng] = samp_prio_momegalx(gdat.trueminmradigalx, self.ng)
                self.cntpback = gdat.cntpback
            
            def log_prior_moments(self, galaxies):
                xx, xy, yy = galaxies[[gdat.indxxxmo, gdat.indxxymo, gdat.indxyymo],:]
                slope = 4.
                a, theta, phi = from_moments(xx, xy, yy)
                u = np.cos(theta)
                return np.log(slope-1) + (slope-1)*np.log(gdat.trueminmradigalx) - slope*np.log(a) - 5*np.log(a) - np.log(u*u) - np.log(1-u*u)
        
            
            def run_sampler(self, gdat, chan, temperature, jj):
                
                if gdat.verbtype > 1:
                    print 'Sample %d started.' % jj
                
                t0 = time.clock()
                
                if gdat.booltile:
                    self.offsxpos = np.random.randint(gdat.sizeregi)
                    self.offsypos = np.random.randint(gdat.sizeregi)
                else:
                    self.offsxpos = 0
                    self.offsypos = 0

                if gdat.verbtype > 2:
                    print 'self.offsxpos'
                    print self.offsxpos
                    print 'self.offsypos'
                    print self.offsypos
                    print 'gdat.sizeregi'
                    print gdat.sizeregi
                
                if gdat.booltile:
                    self.nregx = gdat.sizeimag[0] / gdat.sizeregi + 1
                    self.nregy = gdat.sizeimag[1] / gdat.sizeregi + 1
                else:
                    self.nregx = 1
                    self.nregy = 1
                
                if gdat.verbtype > 1:
                    print 'self.n'
                    print self.n
                
                if gdat.verbtype > 2:
                    print 'self.nregx'
                    print self.nregx
                    print 'self.nregy'
                    print self.nregy
                
                cntpresi = gdat.cntpdata.copy() # residual for zero image is data
                if gdat.strgmodl == 'star':
                    xposeval = self.stars[gdat.indxxpos, :self.n]
                    yposeval = self.stars[gdat.indxypos, :self.n]
                    fluxeval = self.stars[gdat.indxflux, :self.n]
                else:
                    xposphon, yposphon, specphon = retr_tranphon(gdat.gridphon, gdat.amplphon, self.galaxies[:,0:self.ng])
                    print 'self.stars[gdat.indxxpos, :self.n]'
                    summgene(self.stars[gdat.indxxpos, :self.n])
                    print 'xposphon'
                    summgene(xposphon)
                    print 'specphon'
                    summgene(specphon)
                    if gdat.numbener > 1:
                        specphon = retr_spec(gdat, specphon, gdat.colrgalx)
                    print
                    xposeval = np.concatenate([self.stars[gdat.indxxpos, :self.n], xposphon]).astype(np.float32)
                    yposeval = np.concatenate([self.stars[gdat.indxypos, :self.n], yposphon]).astype(np.float32)
                    fluxeval = np.concatenate([self.stars[gdat.indxflux, :self.n], specphon], axis=2).astype(np.float32)
                numbphon = xposeval.size
                
                if gdat.diagmode and jj > 0:
                    gdat.chi2last = np.copy(gdat.chi2stat)

                cntpmodl, gdat.chi2stat = eval_modl(gdat, xposeval, yposeval, fluxeval, self.cntpback, \
                                                                   weig=gdat.weig, cntprefr=cntpresi, \
                                                                   clib=gdatnotp.clibeval, \
                                                                   offsxpos=self.offsxpos, offsypos=self.offsypos)
                llik = -0.5 * gdat.chi2stat
                
                if gdat.diagmode and jj > 0:
                    if np.amax(np.abs(gdat.chi2last - gdat.chi2stat)) > 0.2:
                        print 'Warning! Chi2 state changed during loops!'
                        print 'gdat.cntpdata'
                        summgene(gdat.cntpdata)
                        print 'gdat.chi2last'
                        print gdat.chi2last
                        print 'gdat.chi2stat'
                        print gdat.chi2stat
                        print
                        #raise Exception('')

                #print 'gdat.chi2stat'
                #print gdat.chi2stat
                #print 'llik'
                #print llik
    
                if gdat.verbtype > 1:
                    print 'cntpmodl'
                    summgene(cntpmodl)
                    print 'gdat.chi2stat'
                    summgene(gdat.chi2stat)
                    print 'llik'
                    summgene(llik)
                    print 'gdat.cntpdata'
                    summgene(gdat.cntpdata)
                    print 'cntpresi'
                    summgene(cntpresi)
                    print 
                
                cntpresi -= cntpmodl
                
                # log
                numbpara = retr_numbpara(gdat, self.n, self.ng)
                gdat.chi2totl = np.sum(gdat.weig.astype(np.float64) * (gdat.cntpdata - cntpmodl).astype(np.float64) ** 2)

                if jj % gdat.factthinplot == 0:
                    
                    numbdoff = gdat.numbdata - numbpara
                    chi2doff = gdat.chi2totl / numbdoff
                    if gdat.diagmode:
                        if chi2doff < 0.:
                            print 'Chi2 per dof went negative.'
                            print 'gdat.cntpdata'
                            summgene(gdat.cntpdata)
                            print 'self.n'
                            print self.n
                            print 'numbdoff'
                            print numbdoff
                            print 'gdat.numbdata'
                            print gdat.numbdata
                            print 'numbpara'
                            print numbpara
                            raise Exception('')
                    fmtstr = '\t(all) %0.3f (P) %0.3f (B-D) %0.3f (M-S) %0.3f (Pg) %0.3f (BDg) %0.3f (S-g) %0.3f (gSg) %0.3f (gMS) %0.3f'
                    print 'Sample %d' % jj
                    print 'temp: ', temperature, ', numbstar: ', self.n, ', numbgalx: ', self.ng, ', numbphon: ', numbphon, \
                                                             ', chi2totl: ', gdat.chi2totl, ', numbpara: ', numbpara, ', chi2totldoff: ', chi2doff
                
                # assertions
                if gdat.diagmode:
                    if not np.isfinite(cntpresi).all():
                        print 'cntpresi'
                        summgene(cntpresi)
                        raise Exception('')

                # frame plots
                if (gdat.boolplotsave) and jj % gdat.factthinplot == 0:
                    
                    if gdat.verbtype > 1:
                        print 'Making frame plots...'
                        
                    for i in gdat.indxener:
                        for t in gdat.indxtime:
                            
                            if not gdat.boolframplotenti and (i != 0 or t != 0):
                                break
                            
                            # count map
                            #figr, axis = plt.subplots(figsize=(20, 20))
                            figr, axis = plt.subplots()
                            tick, labl, vmin, vmax = retr_cbar(gdat, 'data')
                            imagscal = retr_imagscal(gdat, gdat.cntpdata[i, :, :, t])
                            
                            imag = axis.imshow(imagscal, origin='lower', interpolation='nearest', cmap='Greys_r', vmin=vmin, vmax=vmax)
                            ## overplot point sources
                            supr_catl(gdat, axis, i, t, self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.stars[gdat.indxflux[i, t], 0:self.n])
                            if gdat.strgmodl == 'galx':
                                supr_catl(gdat, axis, i, t, self.galaxies[gdat.indxxpos, 0:self.ng], self.galaxies[gdat.indxypos, 0:self.ng], self.galaxies[gdat.indxflux[i, t], 0:self.ng], colrmodl='purple')
                            ## limits
                            setp_imaglimt(gdat, axis)
                            cbaxes = figr.add_axes([0.83, 0.1, 0.03, 0.8]) 
                            cbar = figr.colorbar(imag, cax=cbaxes) 
                            cbar.set_ticks(tick)
                            cbar.set_ticklabels(labl)
                            plt.tight_layout()
                            plt.savefig(gdat.pathdatartag + '%s_cntpdata%04d%04d_fram%04d.' % (gdat.rtag, i, t, jj) + gdat.strgplotfile)
                            plt.close()

                            # point source map
                            figr, axis = plt.subplots()
                            tick, labl, vmin, vmax = retr_cbar(gdat, 'data')
                            imagscal = retr_imagscal(gdat, cntpmodl[i, :, :, t] - gdat.cntpback[i, :, :, t])
                            imag = axis.imshow(imagscal, origin='lower', interpolation='nearest', cmap='Greys_r', vmin=vmin, vmax=vmax)
                            ## overplot point sources
                            supr_catl(gdat, axis, i, t, self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.stars[gdat.indxflux[i, t], 0:self.n])
                            if gdat.strgmodl == 'galx':
                                supr_catl(gdat, axis, i, t, self.galaxies[gdat.indxxpos, 0:self.ng], self.galaxies[gdat.indxypos, 0:self.ng], self.galaxies[gdat.indxflux[i, t], 0:self.ng], colrmodl='purple')
                            
                            setp_imaglimt(gdat, axis)
                            cbaxes = figr.add_axes([0.8, 0.1, 0.03, 0.8]) 
                            cbar = figr.colorbar(imag, cax=cbaxes) 
                            cbar.set_ticks(tick)
                            cbar.set_ticklabels(labl)
                            plt.tight_layout()
                            plt.savefig(gdat.pathdatartag + '%s_cntppnts%04d%04d_fram%04d.' % (gdat.rtag, i, t, jj) + gdat.strgplotfile)
                            plt.close()
                            
                            # residual map
                            figr, axis = plt.subplots()
                            tick, labl, vmin, vmax = retr_cbar(gdat, 'resi')
                            imagscal = retr_imagscal(gdat, cntpresi[i, :, :, t])
                            imag = axis.imshow(imagscal, origin='lower', interpolation='nearest', cmap=gdat.cmapresi, vmin=vmin, vmax=vmax)
                            ## overplot point sources
                            supr_catl(gdat, axis, i, t, self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.stars[gdat.indxflux[i, t], 0:self.n])
                            if gdat.strgmodl == 'galx':
                                supr_catl(gdat, axis, i, t, self.galaxies[gdat.indxxpos, 0:self.ng], self.galaxies[gdat.indxypos, 0:self.ng], self.galaxies[gdat.indxflux[i, t], 0:self.ng], colrmodl='purple')
                            
                            setp_imaglimt(gdat, axis)
                            cbaxes = figr.add_axes([0.8, 0.1, 0.03, 0.8]) 
                            cbar = figr.colorbar(imag, cax=cbaxes) 
                            cbar.set_ticks(tick)
                            cbar.set_ticklabels(labl)
                            plt.tight_layout()
                            plt.savefig(gdat.pathdatartag + '%s_cntpresi%04d%04d_fram%04d.' % (gdat.rtag, i, t, jj) + gdat.strgplotfile)
                            plt.close()
                            
                    ## flux histogram
                    plot_fluxhist(gdat, self.stars[gdat.indxflux, 0:self.n], 0, 0, jj=jj)

                if gdat.diagmode:
                    if gdat.chi2totl < 0.:
                        raise Exception('')

                if gdat.verbtype > 1:
                    print 'Starting the loops...'
                if gdat.verbtype > 2:
                    print 'self.stars'
                    summgene(self.stars)

                for a in xrange(gdat.numbloop):

                    if gdat.verbtype > 1:
                        print 'Loop %d started.' % a 

                    t1 = time.clock()
                    self.thisindxproptype = np.random.choice(np.arange(gdat.probprop.size, dtype=int), p=gdat.probprop)

                    # temp -- put this under diagmode
                    regifact = np.zeros((gdat.numbregiyaxi, gdat.numbregixaxi))
                    
                    if gdat.diagmode:
                        for i in gdat.indxener:
                            for t in gdat.indxtime:
                                if (self.stars[gdat.indxflux[i, t], :self.n] < gdat.fittminmflux).any():
                                    print 'gdat.fittminmflux'
                                    print gdat.fittminmflux
                                    print 'self.n'
                                    print self.n
                                    print 'self.stars[gdat.indxflux[i, t], :self.n]'
                                    print self.stars[gdat.indxflux[i, t], :self.n]
                                    raise Exception('')

                    if gdat.verbtype > 1:
                        print 'self.thisindxproptype'
                        print self.thisindxproptype
                        print 'self.n'
                        print self.n
                        print 'gdat.chi2stat'
                        summgene(gdat.chi2stat)
                    
                    gdat.thisindxswlp = jj * gdat.numbloop + a

                    chan['proptype'][gdat.thisindxswlp] = self.thisindxproptype
                    # defaults
                    pn = self.n
                    cntpbackdiff = np.zeros_like(gdat.cntpdata)
                    #if gdat.boolspre:
                    #    cntpbackdiff = np.float32(0.)
                    #else:
                    #    cntpbackdiff = 0.

                    # should regions be perturbed randomly or systematically?
                    if gdat.booltile:
                        self.parity_x = np.random.randint(2)
                        self.parity_y = np.random.randint(2)
                    else:
                        self.parity_x = 0
                        self.parity_y = 0
                    movefns = [self.move_stars, self.birth_death_stars, self.merge_split_stars, self.move_galaxies, self.birth_death_galaxies, self.star_galaxy, \
                                                                                                    self.twostars_galaxy, self.stargalaxy_galaxy, self.merge_split_galaxies]
                    
                    # make the proposal
                    proposal = movefns[self.thisindxproptype]()
                    
                    chan['dt01'][gdat.thisindxswlp] = time.clock() - t1
        
                    if proposal.goodmove:
                        t2 = time.clock()
                        cntpmodldiff, gdat.chi2prop = eval_modl(gdat, proposal.xphon, proposal.yphon, proposal.fphon, cntpbackdiff, \
                               weig=gdat.weig, cntprefr=cntpresi, \
                               clib=gdatnotp.clibeval, \
                               offsxpos=self.offsxpos, offsypos=self.offsypos)
                        llikprop = -0.5 * gdat.chi2prop
                        chan['dt02'][gdat.thisindxswlp] = time.clock() - t2
        
                        t3 = time.clock()
                        
                        #if gdat.booltile:
                            
                        refx, refy = proposal.get_ref_xy()
                        regionx = get_region(refx, self.offsxpos, gdat.sizeregi)
                        regiony = get_region(refy, self.offsypos, gdat.sizeregi)
                        
                        if gdat.verbtype > 1:
                            print 'gdat.chi2prop'
                            summgene(gdat.chi2prop)
                        if gdat.verbtype > 2:
                            print 'refx'
                            summgene(refx)
                            print 'refy'
                            summgene(refy)
                            print 'self.offsxpos'
                            print self.offsxpos
                            print 'self.offsypos'
                            print self.offsypos
                        
                        #else:
                        #    regionx = np.zeros(self.n, dtype=int)
                        #    regiony = np.zeros(self.n, dtype=int)
                        ###
                        '''
                        if i == 0:
                            yy, xx = np.mgrid[0:100,0:100]
                            rxx = get_region(xx, self.offsxpos, sizeregi) % 2
                            ryy = get_region(yy, self.offsypos, sizeregi) % 2
                            rrr = rxx*2 + ryy
                            plt.figure(2)
                            plt.imshow(rrr, interpolation='nearest', origin='lower', cmap='Accent')
                            plt.tight_layout()
                            plt.savefig('region-'+str(int(time.clock()*10))+'.pdf')
                            plt.figure(3)
                            vmax=np.max(np.abs(cntpmodldiff))
                            plt.imshow(cntpmodldiff, interpolation='nearest', origin='lower', cmap='bwr', vmin=-vmax, vmax=vmax)
                            plt.tight_layout()
                            plt.savefig('cntpmodldiff-'+str(int(time.clock()*10))+'.pdf')
                        '''
                        ###
                        
                        if gdat.booltile:
                            llikprop[(1-self.parity_y)::2,:] = float('-inf') # don't accept off-parity regions
                            llikprop[:,(1-self.parity_x)::2] = float('-inf')
                        
                        #if not (gdat.booltile and not np.isfinite(llikprop[1, 1])):
                        #    print 'llikprop - llik'
                        #    print llikprop - llik
                        #    print 
                        
                        dlogP = (llikprop - llik) / temperature
                        if proposal.factor is not None:
                            regifact[regiony, regionx] += proposal.factor
                            dlogP += regifact
                        
                        chan['deltllik'][:, :, gdat.thisindxswlp] = llikprop - llik
                        chan['stdvposiprop'][gdat.thisindxswlp] = gdat.stdvposiprop
                        chan['stdvfluxprop'][gdat.thisindxswlp] = gdat.stdvfluxprop
                        
                        if gdat.verbtype > 1:
                            print 'llikprop'
                            print llikprop
                            print 'llik'
                            print llik
                            print 'dlogP'
                            print dlogP
                        boolregiacpt = (np.log(np.random.uniform(size=(self.nregy, self.nregx))) < dlogP).astype(np.int32)
                        if gdat.verbtype > 1:
                            print 'boolregiacpt'
                            print boolregiacpt
                        acceptprop = boolregiacpt[regiony, regionx]
                        
                        if gdat.diagmode:
                            if self.thisindxproptype == 0 and gdat.stdvfluxprop < 1e-10 and gdat.stdvposiprop < 1e-10 and np.count_nonzero(boolregiacpt) == 0:
                                print 'Proposal not accepted despite small proposal scale.'
                                print 'gdat.stdvfluxprop'
                                print gdat.stdvfluxprop
                                print 'gdat.stdvposiprop'
                                print gdat.stdvposiprop
                                print 'boolregiacpt'
                                print boolregiacpt
                                #raise Exception('')
                        # only keep cntpmodldiff in accepted regions+margins
                        for i in gdat.indxener:
                            for t in gdat.indxtime:
                                cntpmodldiffacpt = np.zeros_like(cntpmodldiff[i, :, :, t])
                                #cntpmodldiffacpt = np.ascontiguousarray(np.zeros_like(cntpmodldiff[i, :, :, t]))
                                gdatnotp.clib.clib_updt_modl(gdat.sizeimag[0], gdat.sizeimag[1], cntpmodldiff[i, :, :, t], \
                                #gdatnotp.clib.clib_updt_modl(gdat.sizeimag[0], gdat.sizeimag[1], np.ascontiguousarray(cntpmodldiff[i, :, :, t]), \
                                                                                                        cntpmodldiffacpt, boolregiacpt, gdat.sizeregi, \
                                                                                                               gdat.marg, self.offsxpos, self.offsypos, gdat.booltile)
                                #print 'it'
                                #print i, t
                                #print 'cntpmodldiffacpt'
                                #summgene(cntpmodldiffacpt)
                                #print 'np.sum((cntpresi[i, :, :, t] - cntpmodldiffacpt)**2 * weig[i, :, :, t])'
                                #print np.sum((cntpresi[i, :, :, t] - cntpmodldiffacpt)**2 * weig[i, :, :, t])
                                # using this cntpmodldiff containing only accepted moves, update chi2proptemp
                                
                                #gdatnotp.clib.clib_eval_llik(gdat.sizeimag[0], gdat.sizeimag[1], cntpmodldiffacpt, cntpresi[i, :, :, t], weig[i, :, :, t], chi2proptemp, sizeregi, \
                                #                                                                                          gdat.marg, self.offsxpos, self.offsypos, gdat.booltile)
                                #print 'new chi2proptemp'
                                #print np.sum(chi2proptemp)
                                #print 'np.sum((gdat.cntpdata[i, :, :, t] - cntpmodl[i, :, :, t])**2 * weig[i, :, :, t])'
                                #print np.sum((gdat.cntpdata[i, :, :, t] - cntpmodl[i, :, :, t])**2 * weig[i, :, :, t])
                                
                                # temp
                                #chi2proptemp = np.array([[np.sum((cntpmodldiffacpt - cntpresi[i, :, :, t])**2 * weig[i, :, :, t])]], dtype=np.float64)
                                
                                cntpresi[i, :, :, t] -= cntpmodldiffacpt # has to occur after clib_eval_llik, because cntpresi is used as cntprefr
                                cntpmodl[i, :, :, t] += cntpmodldiffacpt


                                #print 'chi2proptemp'
                                #summgene(chi2proptemp)
                                #print 'np.sum(chi2proptemp)'
                                #print np.sum(chi2proptemp)
                                
                                #chi2prop += chi2proptemp
                        
                        #print 'chi2prop'
                        #print chi2prop
                        #print 'np.sum((cntpresi - cntpmodldiffacpt)**2 * weig)'
                        ##print np.sum((cntpresi - cntpmodldiffacpt)**2 * weig)
                        ##print 'np.sum((cntpdata - cntpmodl)**2 * weig)'
                        #print np.sum((cntpdata - cntpmodl)**2 * weig)
                        #print 'before: chi2'
                        #print chi2
                        if np.count_nonzero(boolregiacpt) > 0:
                            gdat.chi2copy = np.copy(gdat.chi2stat)
                            for v in range(gdat.numbregiyaxi):
                                for u in range(gdat.numbregixaxi):
                                    if boolregiacpt[v, u] > 0:
                                        gdat.chi2copy[v, u] = gdat.chi2prop[v, u]
                            gdat.chi2stat = gdat.chi2copy
                        
                        # calculate llik
                        llik = -0.5 * gdat.chi2stat
            
                        #print 'llik'
                        #print llik

                        if gdat.verbtype > 1:
                            print 'Implementing accepted proposals...'

                        # implement accepted moves
                        if proposal.idx_move is not None:
                            starsp = proposal.starsp.compress(acceptprop, axis=1)
                            idx_move_a = proposal.idx_move.compress(acceptprop)
                            self.stars[:, idx_move_a] = starsp
                            
                            if gdat.diagmode:
                                if np.unique(proposal.starsp[gdat.indxflux, :]).size != proposal.starsp[gdat.indxflux, :].size and proposal.starsp[gdat.indxflux, :].size > 1:
                                    print 'Proposed flux not all unique'
                                    print 'proposal.stars0[gdat.indxflux, :]'
                                    print proposal.stars0[gdat.indxflux, :]
                                    print 'proposal.starsp[gdat.indxflux, :]'
                                    print proposal.starsp[gdat.indxflux, :]
                                    #raise Exception('')
                                    
                            if gdat.verbtype > 1:
                                print 'proposal.idx_move is not None'
                                print 'proposal.starsp'
                                summgene(proposal.starsp)
                                print 'proposal.starsp[gdat.indxflux, :]'
                                print proposal.starsp[gdat.indxflux, :]
                                print 'idx_move_a'
                                summgene(idx_move_a)
                                print 'New self.stars'
                                summgene(self.stars)
                            
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
                            # maxmnumbstar is correct, not n, because x,y,f are full maxmnumbstar arrays
                            self.stars[:, 0:gdat.maxmnumbstar-num_kill] = np.delete(self.stars, idx_kill_a, axis=1)
                            self.stars[:, gdat.maxmnumbstar-num_kill:] = 0
                            self.n -= num_kill
                        if proposal.idx_kill_g is not None:
                            idx_kill_a = proposal.idx_kill_g.compress(acceptprop)
                            num_kill = idx_kill_a.size
                            # like above, ngalx is correct
                            self.galaxies[:, 0:gdat.maxmnumbgalx-num_kill] = np.delete(self.galaxies, idx_kill_a, axis=1)
                            self.galaxies[:, gdat.maxmnumbgalx-num_kill:] = 0
                            self.ng -= num_kill
                        chan['dt03'][gdat.thisindxswlp] = time.clock() - t3
        
                        chan['accp'][:, :, gdat.thisindxswlp] = boolregiacpt
                    else:
                        chan['booloutb'][gdat.thisindxswlp] = 1
                    
                    if gdat.diagmode:
                        if proposal.factor is not None and not np.isfinite(proposal.factor).any():
                            print 'Warning! proposal.factor is not finite!'
                            #raise Exception('')
                    if proposal.factor is not None and proposal.factor != []:
                        chan['lposterm'][gdat.thisindxswlp, :, :] = regifact

                    if gdat.diagmode:
                        gdat.chi2prev = np.copy(gdat.chi2stat)
                   
                    gdat.numbswepaccplogg = 100
                    if jj % gdat.factthinplot == 0:
                        maxmsweplogg = jj * gdat.numbloop + a
                        minmsweplogg = max(0, maxmsweplogg - gdat.numbswepaccplogg)
                        indxsweplogg = np.arange(minmsweplogg, maxmsweplogg)
                    
                    if gdat.verbtype > 1:
                        print 'gdat.chi2stat'
                        summgene(gdat.chi2stat)
                        print 'llik'
                        summgene(llik)
                        if proposal.goodmove:
                            for i in gdat.indxener:
                                for t in gdat.indxtime:
                                    print 'cntpmodldiff[i, :, :, t]'
                                    summgene(cntpmodldiff[i, :, :, t])
                                    #print 'cntpmodldiffacpt[i, :, :, t]'
                                    #summgene(cntpmodldiffacpt[i, :, :, t])
                        print 'Loop %d ended.' % a
                        print
                        print
                        print

                if jj % gdat.factthinplot == 0:
                    chan['dt01'] *= 1e3
                    chan['dt02'] *= 1e3
                    chan['dt03'] *= 1e3
                    maxmsweplogg = jj * gdat.numbloop + a
                    minmsweplogg = max(0, maxmsweplogg - 1000)
                    indxsweplogg = np.arange(minmsweplogg, maxmsweplogg)
                    statarrays = [np.mean(np.mean(chan['accp'], axis=0), axis=0), chan['booloutb'], chan['dt01'], chan['dt02'], chan['dt03']]
                    
                    for j in gdat.indxvarbdiag:
                        print gdat.strgvarbdiag[j] + '\t(all) %0.3f' % np.mean(statarrays[j][indxsweplogg]),
                        for k in xrange(len(gdat.listproptype)):
                            print '(' + gdat.listproptype[k] + ') %0.3f' % np.mean(statarrays[j][indxsweplogg][chan['proptype'][indxsweplogg] == k]),
                        print
                        if j == 1:
                            print '-' * 16
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
        
                return self.n, self.ng, gdat.chi2totl
        
            
            def idx_parity_stars(self):
                return idx_parity(self.stars[gdat.indxxpos,:], self.stars[gdat.indxypos,:], self.n, self.offsxpos, self.offsypos, self.parity_x, self.parity_y, gdat.sizeregi)
        
            
            def idx_parity_galaxies(self):
                return idx_parity(self.galaxies[gdat.indxxpos,:], self.galaxies[gdat.indxypos,:], self.ng, self.offsxpos, self.offsypos, self.parity_x, self.parity_y, gdat.sizeregi)
        
            
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
                
                if gdat.verbtype > 1:
                    print 'move_stars()'

                idx_move = self.idx_parity_stars()
                nw = idx_move.size
                stars0 = self.stars.take(idx_move, axis=1)
                if gdat.verbtype > 1:
                    print 'idx_move'
                    summgene(idx_move)
                    print 'stars0'
                    summgene(stars0)
                starsp = np.empty_like(stars0)
                if gdat.priotype == 'info':
                    f0 = stars0[gdat.indxflux[0, 0], :]
                else:
                    f0 = stars0[gdat.indxflux, :]
                
                if gdat.diagmode:
                    if (f0 < gdat.fittminmflux).any():
                        print 'Flux went below minimum'
                        raise Exception('')

                psf0 = gdat.cntppsfnusam[0,:gdat.numbsidepsfnusam:gdat.factusam, :gdat.numbsidepsfnusam:gdat.factusam]
                stdf = 1./np.sqrt(np.median(gdat.weig[0,:,:,0]) * np.sum(psf0*psf0)) # assumes single band
                nest = 0.5 * gdat.maxmnumbstar / (gdat.numbregixaxi * gdat.numbregiyaxi)
                lindf = stdf / np.sqrt(nest * (2+gdat.numbener)) # put numbtime here? 60
                logdf = 0.01 / np.sqrt(nest * (2+gdat.numbener))
                if gdat.boolspre:
                    lindf = np.float32(lindf)
                    logdf = np.float32(logdf)
                ff = np.log(logdf**2 * f0 + logdf * np.sqrt(lindf**2 + logdf**2 * f0**2)) / logdf
                ffmin = np.log(logdf*logdf*gdat.fittminmflux + logdf*np.sqrt(lindf*lindf + logdf*logdf*gdat.fittminmflux*gdat.fittminmflux)) / logdf
                if gdat.boolspre:
                    dff = gdat.stdvfluxprop * np.random.normal(size=nw*gdat.numbflux).astype(np.float32).reshape((gdat.numbener, gdat.numbtime, nw))
                else:
                    dff = gdat.stdvfluxprop * np.random.normal(size=nw*gdat.numbflux).reshape((gdat.numbener, gdat.numbtime, nw))
                fluxabov = ff - ffmin
                boolfluxoutb = -dff > fluxabov
                dff[boolfluxoutb] = -2. * fluxabov[boolfluxoutb] - dff[boolfluxoutb]
                pff = ff + dff
                explogdfpff = np.exp((logdf*pff).astype(np.float64)).astype(np.float32)
                pf = 1./explogdfpff * (-lindf*lindf*logdf*logdf + explogdfpff**2) / (2*logdf*logdf)
                # calculate flux distribution prior factor
                dlogf = np.log(pf/f0)
                # assumes flat priors over colors
                # temp
                if gdat.priotype == 'info':
                    factor = -gdat.fittfluxdistslop * dlogf[0, 0, :]
                else:
                    factor = -gdat.fittfluxdistslop * np.sum(dlogf)
                
                #print 'dlogf'
                #summgene(dlogf)
                #print 'factor'
                #print factor
                #print

                #if False and 
                if gdat.verbtype > 2:
                    print 'f0'
                    print f0
                    print 'lindf'
                    print lindf
                    print 'logdf'
                    print logdf
                    #print 'ff'
                    #print ff
                    print 'ffmin'
                    print ffmin
                    print 'fluxabov'
                    summgene(fluxabov)
                    print 'boolfluxoutb'
                    print boolfluxoutb
                    print 'dff'
                    print dff
                    print 'pff'
                    print pff
                    print 'dlogf'
                    print dlogf
                    print 'pf'
                    summgene(pf)
                    print pf
                    print 'factor'
                    print factor
                
                if gdat.diagmode:
                    if (pf < gdat.fittminmflux).any():
                        print 'pf'
                        print pf
                        indx = np.where(pf < gdat.fittminmflux)
                        print 'dff[indx]'
                        print dff[indx]
                        print 'fluxabov[indx]'
                        print fluxabov[indx]
                        print 'boolfluxoutb[indx]'
                        print boolfluxoutb[indx]
                        print 'pf[indx]'
                        print pf[indx]
                        print 'gdat.fittminmflux'
                        print gdat.fittminmflux
                        if gdat.boolspre:
                            print 'Warning! Proposed flux is lower than the minimum flux!'
                        else:
                            pass
                            #raise Exception('')

                if gdat.diagmode and (pf < gdat.fittminmflux).any():
                    print 'proposed flux went below minimum'
                    print f0 > gdat.fittminmflux
                    raise Exception("")
                
                if not gdat.strgmode == 'forc':
                    psfx = gdat.cntppsfnusam[0, 0:gdat.numbsidepsfnusam:gdat.factusam, 1:gdat.numbsidepsfnusam:gdat.factusam]
                    dpdx = (psfx - psf0) * gdat.factusam
                    #print 'rms', 1./ np.sqrt(np.median(gdat.weig[0,:,:,0]) * np.sum(dpdx*dpdx) * nest * gdat.numbener ), 'old', gdat.stdvposiprop                    
                    dpos_rms = 1./ np.sqrt(np.median(gdat.weig[0,:,:,0]) * np.sum(dpdx*dpdx) * nest * (2+gdat.numbener) ) / \
                            np.amax(np.amax(np.maximum(f0, pf), axis=0), axis=0)
                    #dpos_rms = gdat.stdvposiprop / np.amax(np.amax(np.maximum(f0, pf), axis=0), axis=0)
                    if gdat.stdvposiprop == 1e-100:
                        print 'Not lower-bounding the positional position size at 1e-3...'
                        print
                    else:
                        dpos_rms[dpos_rms < 1e-3] = 1e-3
                    
                    if gdat.boolspre:
                        dx = np.random.normal(size=nw).astype(np.float32)*dpos_rms
                        dy = np.random.normal(size=nw).astype(np.float32)*dpos_rms
                    else:
                        dx = np.random.normal(size=nw)*dpos_rms
                        dy = np.random.normal(size=nw)*dpos_rms
                    
                    if gdat.verbtype > 1:
                        print 'dpos_rms'
                        summgene(dpos_rms)
                        print 'dx'
                        summgene(dx)
                        print 'dy'
                        summgene(dy)
                        print
                    
                    starsp[gdat.indxxpos,:] = stars0[gdat.indxxpos,:] + dx
                    starsp[gdat.indxypos,:] = stars0[gdat.indxypos,:] + dy
                else:
                    starsp[gdat.indxxpos,:] = stars0[gdat.indxxpos,:]
                    starsp[gdat.indxypos,:] = stars0[gdat.indxypos,:]
                starsp[gdat.indxflux, :] = pf
                self.bounce_off_edges(starsp)

                
                if gdat.verbtype > 1:
                    print 'stars0[gdat.indxflux]'
                    summgene(stars0[gdat.indxflux])
                    print 'starsp'
                    summgene(starsp)
                    print 'starsp[gdat.indxflux]'
                    summgene(starsp[gdat.indxflux])
        
                proposal = Proposal()
                proposal.add_move_stars(idx_move, stars0, starsp)
                proposal.set_factor(factor)
                return proposal
        
            
            def birth_death_stars(self):
                lifeordeath = np.random.randint(2)
                if gdat.booltile:
                    nbd = (self.nregx * self.nregy) / 4
                else:
                    nbd = 1
                proposal = Proposal()
                
                # birth
                if self.n == 0 or lifeordeath and self.n < gdat.maxmnumbstar: # need room for at least one source
                    nbd = min(nbd, gdat.maxmnumbstar - self.n) # birth nbd stars, or however many there's room for
                    # want number of regions in each direction, divided by two, rounded up
                    mregx = ((gdat.sizeimag[0] / gdat.sizeregi + 1) + 1) / 2 # assumes that sizeimag are multiples of sizeregi
                    mregy = ((gdat.sizeimag[1] / gdat.sizeregi + 1) + 1) / 2
                    if gdat.boolspre:
                        starsb = np.empty((gdat.numbparastar, nbd), dtype=np.float32)
                    else:
                        starsb = np.empty((gdat.numbparastar, nbd))
                    starsb[gdat.indxxpos,:] = (np.random.randint(mregx, size=nbd)*2 + self.parity_x + np.random.uniform(size=nbd))*gdat.sizeregi - self.offsxpos
                    starsb[gdat.indxypos,:] = (np.random.randint(mregy, size=nbd)*2 + self.parity_y + np.random.uniform(size=nbd))*gdat.sizeregi - self.offsypos

                    starsb[gdat.indxflux, :] = gdat.fittminmflux * \
                                        np.exp(np.random.exponential(scale=1./(gdat.fittfluxdistslop-1.), size=nbd*gdat.numbflux)).reshape((gdat.numbflux, nbd))

                    # temp -- entimodi
                    #colr = np.empty((gdat.numbener - 1, nbd))
                    #lcpr = np.empty((gdat.numbtime - 1, nbd))
                    #if gdat.numbener > 1:
                    #    for i in range(gdat.numbener - 1):
                    #        colr[i, :] = np.random.normal(loc=gdat.meancolr[i], scale=gdat.stdvcolr[i], size=nbd)
                    #

                    #if gdat.numbtime > 1:
                    #    for t in range(gdat.numbtime - 1):
                    #        lcpr[t, :] = np.random.normal(loc=0., scale=gdat.stdvlcpr, size=nbd)
                    #for i in gdat.indxener:
                    #    for t in gdat.indxtime:
                    #        if i == 0 and t == 0:
                    #            starsb[gdat.indxflux[0, 0], :] = gdat.fittminmflux * np.exp(np.random.exponential(scale=1./(gdat.fittfluxdistslop-1.),size=nbd))
                    #        else:
                    #            starsb[gdat.indxflux[i, t], :] = starsb[gdat.indxflux[0, 0], :] 
                    #            if gdat.numbener > 1:
                    #                starsb[gdat.indxflux[i, t], :] *= 10**(0.4*colr[i-1])# * nmgy_per_count[0] / nmgy_per_count[i] 
                    #            if gdat.numbtime > 1:
                    #                starsb[gdat.indxflux[i, t], :] *= lcpr[t-1]
        
                    # some sources might be generated outside image
                    inbounds = self.in_bounds(starsb)
                    starsb = starsb.compress(inbounds, axis=1)
                    factor = np.full(starsb.shape[1], -self.penalty)
                    proposal.add_birth_stars(starsb)
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
                    else:
                        factor = float('-inf')
                else:
                    factor = float('-inf')
                
                proposal.set_factor(factor)

                return proposal
        
            
            def merge_split_stars(self):
                
                boolsplt = np.random.randint(2)
                idx_reg = self.idx_parity_stars()
                fluxspmrtotl = 0
                low_n = 0
                # temp -- not consistent with tiling
                if gdat.priotype == 'prio':
                    idx_bright = idx_reg[np.where(self.stars[gdat.indxflux[0, 0], idx_reg] > 2. * gdat.fittminmflux)]
                else:
                    idx_bright = idx_reg[np.where((self.stars[gdat.indxflux, :] > 2. * gdat.fittminmflux).all(axis=0).all(axis=0))]
                numbbrgt = idx_bright.size
                
                if gdat.booltile:
                    numbspmr = (self.nregx * self.nregy) / 4
                else:
                    numbspmr = 1
                
                if gdat.verbtype > 1:
                    print 'Deciding to propose a split or merge...'
                    print 'numbbrgt'
                    print numbbrgt
                    print 'numbspmr'
                    print numbspmr

                goodmove = False
                proposal = Proposal()
                # split
                if boolsplt and self.n > 0 and self.n < gdat.maxmnumbstar and numbbrgt > 0: # need something to split, but don't exceed maxmnumbstar
                    numbspmr = min(numbspmr, numbbrgt, gdat.maxmnumbstar - self.n) # need bright source AND room for split source
                    if gdat.boolspre:
                        dx = (np.random.normal(size=numbspmr)*self.scalspmrposi).astype(np.float32)
                        dy = (np.random.normal(size=numbspmr)*self.scalspmrposi).astype(np.float32)
                    else:
                        dx = (np.random.normal(size=numbspmr)*self.scalspmrposi)
                        dy = (np.random.normal(size=numbspmr)*self.scalspmrposi)
                    if gdat.verbtype > 1:
                        print 'Proposing a split...'
                        print 'gdat.maxmnumbstar'
                        print gdat.maxmnumbstar
                        print 'self.n'
                        print self.n
                        print 'dx'
                        print dx
                        print 'dy'
                        print dy
                        print
                    idx_move = np.random.choice(idx_bright, size=numbspmr, replace=False)
                    stars0 = self.stars.take(idx_move, axis=1)
                    if gdat.verbtype > 1:
                        print 'stars0'
                        summgene(stars0)
                    x0 = stars0[gdat.indxxpos, :]
                    y0 = stars0[gdat.indxypos, :]
                    if gdat.priotype == 'info':
                        # temp -- based on 0, 0
                        f0 = stars0[gdat.indxflux[0, 0], :]
                        varbrandtemp = np.random.uniform(size=numbspmr)
                    else:
                        varbrandtemp = np.random.uniform(size=numbspmr * gdat.numbflux).reshape((gdat.numbener, gdat.numbtime, numbspmr))
                        f0 = stars0[gdat.indxflux, :]
                    fluxnormminm = f0 / gdat.fittminmflux
                    
                    frac = (1. / fluxnormminm + varbrandtemp * (1. - 2. / fluxnormminm))
                    
                    if gdat.boolspre:
                        frac = frac.astype(np.float32)
                    
                    # temp
                    starsp = np.empty_like(stars0)
                    if gdat.priotype == 'info':
                        dfrcenti = 1e-3 * np.random.randn(gdat.numbflux * numbspmr).reshape((gdat.numbener, gdat.numbtime, numbspmr))
                        dfrcenti[0, 0, :] = 0.
                        fractemp = frac[0, 0, :]
                    else:
                        fractemp = frac[0, 0]
                    starsp[gdat.indxxpos, :] = x0 + ((1. - fractemp) * dx)
                    starsp[gdat.indxypos, :] = y0 + ((1. - fractemp) * dy)
                    
                    if gdat.priotype == 'info':
                        fracenti = frac * (1. + dfrcenti)
                    
                        # correct for fracenti > 1 or fracenti < 0
                        # temp
                        indxloww = np.where(fracenti < 0.)
                        if indxloww[0].size > 0:
                            fracenti[indxloww] = -fracenti[indxloww]
                        
                        indxhigh = np.where(fracenti > 1.)
                        if indxhigh[0].size > 0:
                            fracenti[indxhigh] -= 2. * (fracenti[indxhigh] - 1.)
                            fracenti[indxbadd] = fracenti[indxbadd] % 1.
                        starsp[gdat.indxflux, :] = f0 * fracenti
                    else:
                        starsp[gdat.indxflux, :] = f0 * frac
                        
                    starsb = np.empty_like(stars0)
                    if gdat.priotype == 'info':
                        starsb[gdat.indxxpos, :] = x0 - fractemp * dx
                        starsb[gdat.indxypos, :] = y0 - fractemp * dy
                    else:
                        starsb[gdat.indxxpos, :] = x0 - frac[0, 0] * dx
                        starsb[gdat.indxypos, :] = y0 - frac[0, 0] * dy

                    if gdat.verbtype > 1:
                        print 'starsb[gdat.indxflux, :]'
                        summgene(starsb[gdat.indxflux, :])
                    
                    if gdat.priotype == 'info':
                        starsb[gdat.indxflux, :] = f0 * (1. - fracenti)
                    else:
                        starsb[gdat.indxflux, :] = f0 * (1. - frac)
                    
                    if gdat.verbtype > 1:
                        print 'f0'
                        summgene(f0)
                        print 'frac'
                        summgene(frac)
                        print 'starsb[gdat.indxflux, :]'
                        summgene(starsb[gdat.indxflux, :])
                        if gdat.priotype == 'info':
                            print 'fracenti'
                            summgene(fracenti)
                    
                    inbounds = np.logical_and(self.in_bounds(starsp), self.in_bounds(starsb))
                    stars0 = stars0.compress(inbounds, axis=1)
                    starsp = starsp.compress(inbounds, axis=1)
                    starsb = starsb.compress(inbounds, axis=1)
                    idx_move = idx_move.compress(inbounds)
                    fluxnormminm = fluxnormminm.compress(inbounds)
                    frac = frac.compress(inbounds, axis=2)
                    numbspmr = idx_move.size
                    goodmove = numbspmr > 0
                    if goodmove:
                        proposal.add_move_stars(idx_move, stars0, starsp)
                        proposal.add_birth_stars(starsb)
        
                    # need to calculate factor
                    fluxspmrtotl = stars0[gdat.indxflux,:]
                    invpairs = np.empty(numbspmr)
                    for k in xrange(numbspmr):
                        xtemp = self.stars[gdat.indxxpos, 0:self.n].copy()
                        ytemp = self.stars[gdat.indxypos, 0:self.n].copy()
                        xtemp[idx_move[k]] = starsp[gdat.indxxpos, k]
                        ytemp[idx_move[k]] = starsp[gdat.indxypos, k]
                        xtemp = np.concatenate([xtemp, starsb[gdat.indxxpos, k:k+1]])
                        ytemp = np.concatenate([ytemp, starsb[gdat.indxypos, k:k+1]])
        
                        invpairs[k] =  1./neighbours(xtemp, ytemp, self.scalspmrposi, idx_move[k]) #divide by zero
                        invpairs[k] += 1./neighbours(xtemp, ytemp, self.scalspmrposi, self.n)
                    invpairs *= 0.5
                    if gdat.verbtype > 1:
                        print 'starsb'
                        print starsb
                        print 'starsp'
                        print starsp
                        print
                    
                    if gdat.diagmode:
                        if boolsplt:
                            if (starsp[gdat.indxflux, :] < gdat.fittminmflux).any():
                                print 'starsp[gdat.indxflux, :]'
                                summgene(starsp[gdat.indxflux, :])
                                print 'gdat.fittminmflux'
                                print gdat.fittminmflux
                                raise Exception('')
                            
                            if gdat.priotype == 'info':
                                if (starsb[gdat.indxflux[0, 0], :] < gdat.fittminmflux).any():
                                    print 'starsb[gdat.indxflux[0, 0], :]'
                                    summgene(starsb[gdat.indxflux[0, 0], :])
                                    print 'gdat.fittminmflux'
                                    print gdat.fittminmflux
                                    raise Exception('')
                                if (starsb[gdat.indxflux[0, 0], :] < gdat.fittminmflux).any():
                                    print 'starsb[gdat.indxflux[0, 0], :]'
                                    summgene(starsb[gdat.indxflux[0, 0], :])
                                    raise Exception('')
                
                # merge
                elif not boolsplt and idx_reg.size > 1: # need two things to merge!
                    
                    if gdat.verbtype > 1:
                        print 'Proposing a merge...'
                        print 'numbspmr'
                    
                    numbspmr = min(numbspmr, idx_reg.size/2)
                    idx_move = np.empty(numbspmr, dtype=np.int)
                    idx_kill = np.empty(numbspmr, dtype=np.int)
                    choosable = np.zeros(gdat.maxmnumbstar, dtype=np.bool)
                    choosable[idx_reg] = True
                    nchoosable = float(idx_reg.size)
                    invpairs = np.empty(numbspmr)
        
                    for k in xrange(numbspmr):
                        idx_move[k] = np.random.choice(gdat.maxmnumbstar, p=choosable/nchoosable)
                        invpairs[k], idx_kill[k] = neighbours(self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.scalspmrposi, idx_move[k], generate=True)
                        if invpairs[k] > 0:
                            invpairs[k] = 1./invpairs[k]
                        # prevent sources from being involved in multiple proposals
                        if not choosable[idx_kill[k]]:
                            idx_kill[k] = -1
                        if idx_kill[k] != -1:
                            invpairs[k] += 1./neighbours(self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.scalspmrposi, idx_kill[k])
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
        
                    if gdat.priotype == 'info':
                        f0 = stars0[gdat.indxflux[0, 0], :]
                        fk = starsk[gdat.indxflux[0, 0], :]
                    else:
                        f0 = stars0[gdat.indxflux, :]
                        fk = starsk[gdat.indxflux, :]
                    fluxspmrtotl = f0 + fk
                    fluxnormminm = fluxspmrtotl / gdat.fittminmflux
                    
                    if gdat.priotype == 'info':
                        frac = f0 / fluxspmrtotl
                    else:
                        frac = f0[0, 0] / fluxspmrtotl[0, 0]
                    
                    if gdat.priotype == 'info':
                        dfrcenti = (stars0[gdat.indxflux, :] / (stars0[gdat.indxflux, :] + stars0[gdat.indxflux[0, 0], :]) - frac)
                    
                    # temp
                    starsp = np.empty_like(stars0)
                    starsp[gdat.indxxpos,:] = frac * stars0[gdat.indxxpos,:] + (1. - frac) * starsk[gdat.indxxpos,:]
                    starsp[gdat.indxypos,:] = frac * stars0[gdat.indxypos,:] + (1. - frac) * starsk[gdat.indxypos,:]
                    starsp[gdat.indxflux,:] = f0 + fk
                    if goodmove:
                        proposal.add_move_stars(idx_move, stars0, starsp)
                        proposal.add_death_stars(idx_kill, starsk)
                    # turn numbbrgt into an array
                    numbbrgt = numbbrgt - (f0 > 2*gdat.fittminmflux) - (fk > 2*gdat.fittminmflux) + (starsp[gdat.indxflux,:] > 2*gdat.fittminmflux)
                    
                    if gdat.verbtype > 1:
                        print 'starsk'
                        print starsk
                        print 'starsp'
                        print starsp
                        print
                    
                else:
                    if gdat.verbtype > 1:
                        print 'Could not propose a split or merge.'
                
                if goodmove:
                    if gdat.verbtype > 2:
                        print 'numbspmr'
                        print numbspmr
                        print 'frac'
                        summgene(frac)
                        print 'invpairs'
                        summgene(invpairs)
                        print 'fluxspmrtotl'
                        summgene(fluxspmrtotl)
                    factor = np.log(gdat.fittfluxdistslop - 1.) + \
                                                (gdat.fittfluxdistslop - 1.) * np.log(gdat.fittminmflux) + \
                                                -gdat.fittfluxdistslop * np.log(frac * (1. - frac) * fluxspmrtotl) + \
                                                np.log(2. * np.pi * self.scalspmrposi**2) + \
                                                -np.log(gdat.sizeimag[0] * gdat.sizeimag[1]) + \
                                                np.log(1. - 2. / fluxnormminm) + \
                                                np.log(numbbrgt) + \
                                                np.log(invpairs) + \
                                                np.log(fluxspmrtotl)

                    if gdat.priotype == 'info':
                        lprienti = -np.sqrt(2. * np.pi) * 1e-3 * (gdat.numbflux - 1) - 0.5 * np.sum(dfrcenti**2) / 1e-6
                        factor += lprienti 
                    else:
                        factor = np.sum(factor)
                    if not boolsplt:
                        factor *= -1
                        factor += self.penalty
                    else:
                        factor -= self.penalty
                    proposal.set_factor(np.sum(factor))
                   
                    if gdat.diagmode:
                        if factor.size != 1:
                            print 'factor'
                            print factor
                            raise Exception('')
                        if not np.isfinite(factor):
                            print 'Warning! factor is not finite!'
                            print 'self.thisindxproptype'
                            print self.thisindxproptype
                            print 'factor'
                            print factor
                            #raise Exception('')
                return proposal
        
            
            def move_galaxies(self):
                idx_move_g = self.idx_parity_galaxies()
                nw = idx_move_g.size
                galaxies0 = self.galaxies.take(idx_move_g, axis=1)
                f0g = galaxies0[gdat.indxflux,:]
        
                psf0 = gdat.cntppsfnusam[0,:gdat.numbsidepsfnusam:gdat.factusam, :gdat.numbsidepsfnusam:gdat.factusam]
                stdf = 1./np.sqrt(np.median(gdat.weig[0,:,:,0]) * np.sum(psf0*psf0)) # assumes single band
                nest = 0.5 * gdat.maxmnumbstar / (gdat.numbregixaxi * gdat.numbregiyaxi)
                lindf = stdf / np.sqrt(nest * (2+gdat.numbener)) # put numbtime here? 60
                logdf = 0.01 / np.sqrt(nest * (2+gdat.numbener))
                ff = np.log(logdf*logdf*f0g + logdf*np.sqrt(lindf*lindf + logdf*logdf*f0g*f0g)) / logdf
                ffmin = np.log(logdf*logdf*gdat.fittminmfluxgalx + logdf*np.sqrt(lindf*lindf + \
                                                                logdf*logdf*gdat.fittminmfluxgalx*gdat.fittminmfluxgalx)) / logdf
                dff = np.random.normal(size=(gdat.numbener, gdat.numbtime, nw)).astype(np.float32)
                fluxabov = ff - ffmin
                boolfluxoutb = (-dff > fluxabov)
                dff[boolfluxoutb] = -2*fluxabov[boolfluxoutb] - dff[boolfluxoutb]
                pff = ff + dff
                #pfg = np.exp(-logdf*pff) * (-lindf*lindf*logdf*logdf+np.exp(2*logdf*pff)) / (2*logdf*logdf)
       
                explogdfpff = np.exp((logdf*pff).astype(np.float64)).astype(np.float32)
                pfg = 1./explogdfpff * (-lindf*lindf*logdf*logdf + explogdfpff**2) / (2*logdf*logdf)

                dlogfg = np.log(pfg/f0g)
                factor = -np.sum(gdat.fittfluxdistslopgalx*dlogfg, axis=(0, 1))
                 
                psfx = gdat.cntppsfnusam[0, 0:gdat.numbsidepsfnusam:gdat.factusam, 1:gdat.numbsidepsfnusam:gdat.factusam]
                dpdx = (psfx - psf0) * gdat.factusam
                dpos_rms = 1./ np.sqrt(np.median(gdat.weig[0,:,:,0]) * np.sum(dpdx*dpdx) * nest * (2+gdat.numbener) ) / \
                        np.amax(np.amax(np.maximum(f0g, pfg), axis=0), axis=0)
                dxg = np.random.normal(size=nw).astype(np.float32)*dpos_rms
                dyg = np.random.normal(size=nw).astype(np.float32)*dpos_rms
                dxxg = np.random.normal(size=nw, scale=0.1).astype(np.float32)
                dxyg = np.random.normal(size=nw, scale=0.1).astype(np.float32)
                dyyg = np.random.normal(size=nw, scale=0.1).astype(np.float32)
                galaxiesp = np.empty_like(galaxies0)
                xx0g, xy0g, yy0g = galaxies0[[gdat.indxxxmo, gdat.indxxymo, gdat.indxyymo],:]
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
                galaxiesp[gdat.indxxxmo,:] = pxxg
                galaxiesp[gdat.indxxymo,:] = pxyg
                galaxiesp[gdat.indxyymo,:] = pyyg
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
                if lifeordeath and self.ng < gdat.maxmnumbgalx: # need room for at least one source
                    nbd = min(nbd, gdat.maxmnumbgalx-self.ng) # add nbd sources, or just as many as will fit
                                            # mildly violates detailed balance when n close to maxmnumbstar
                    # want number of regions in each direction, divided by two, rounded up
                    mregx = ((gdat.sizeimag[0] / gdat.sizeregi + 1) + 1) / 2 # assumes that sizeimag are multiples of sizeregi
                    mregy = ((gdat.sizeimag[1] / gdat.sizeregi + 1) + 1) / 2
                    galaxiesb = np.empty((6,nbd))
                    galaxiesb[gdat.indxxpos,:] = (np.random.randint(mregx, size=nbd)*2 + self.parity_x + np.random.uniform(size=nbd))*gdat.sizeregi - self.offsxpos
                    galaxiesb[gdat.indxypos,:] = (np.random.randint(mregy, size=nbd)*2 + self.parity_y + np.random.uniform(size=nbd))*gdat.sizeregi - self.offsypos
                    galaxiesb[gdat.indxflux,:] = gdat.fittminmflux * np.exp(np.random.exponential(scale=1./(gdat.fittfluxdistslop-1.),size=nbd))
                    galaxiesb[[gdat.indxxxmo, gdat.indxxymo, gdat.indxyymo],:] = samp_prio_momegalx(gdat.trueminmradigalx, nbd)
        
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
                if starorgalx and self.n > 0 and self.ng < gdat.maxmnumbgalx:
                    idx_reg = self.idx_parity_stars()
                    nsg = min(nsg, min(idx_reg.size, gdat.maxmnumbgalx-self.ng))
                    if nsg > 0:
                        idx_kill = np.random.choice(idx_reg, size=nsg, replace=False)
                        starsk = self.stars.take(idx_kill, axis=1)
                        galaxiesb = np.empty((6, nsg))
                        galaxiesb[[gdat.indxxpos, gdat.indxypos, gdat.indxflux],:] = starsk
                        galaxiesb[[gdat.indxxxmo, gdat.indxxymo, gdat.indxyymo],:] = samp_prio_momegalx(gdat.trueminmradigalx, nsg)
                        factor = np.full(nsg, self.penalty-self.penalty_g) # TODO factor if star and galaxy flux distributions different
                        proposal.add_death_stars(idx_kill, starsk)
                        proposal.add_birth_galaxies(galaxiesb)
                        proposal.set_factor(factor)
                # galaxy -> star
                elif not starorgalx and self.ng > 1 and self.n < gdat.maxmnumbstar:
                    idx_reg = self.idx_parity_galaxies()
                    nsg = min(nsg, min(idx_reg.size, gdat.maxmnumbstar-self.n))
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
                fluxspmrtotl = 0
                low_n = 0
                idx_bright = idx_reg_g.take(np.flatnonzero(self.galaxies[gdat.indxflux, :].take(idx_reg_g) > 2*gdat.fittminmflux)) # in region and bright enough to make two stars
                numbbrgt = idx_bright.size # can only split bright galaxies
        
                numbspmr = (self.nregx * self.nregy) / 4
                goodmove = False
                proposal = Proposal()
                # split
                if boolsplt and self.ng > 0 and self.n < gdat.maxmnumbstar-2 and numbbrgt > 0: # need something to split, but don't exceed maxmnumbstar
                    numbspmr = min(numbspmr, numbbrgt, (gdat.maxmnumbstar-self.n)/2) # need bright galaxy AND room for split stars
                    idx_kill_g = np.random.choice(idx_bright, size=numbspmr, replace=False)
                    galaxiesk = self.galaxies.take(idx_kill_g, axis=1)
                    xkg, ykg, fkg, xxkg, xykg, yykg = galaxiesk
                    fluxnormminm = fkg / gdat.fittminmflux # again, care about fmin for stars
                    frac = (1./fluxnormminm + np.random.uniform(size=numbspmr)*(1. - 2./fluxnormminm)).astype(np.float32)
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
                    fluxnormminm = fluxnormminm.compress(inbounds)
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
                    fluxspmrtotl = fkg
                    weigoverpairs = np.empty(numbspmr) # w (1/sum w_1 + 1/sum w_2) / 2
                    for k in xrange(numbspmr):
                        xtemp = self.stars[gdat.indxxpos, 0:self.n].copy()
                        ytemp = self.stars[gdat.indxypos, 0:self.n].copy()
                        xtemp = np.concatenate([xtemp, starsb[0, k:k+1, gdat.indxxpos], starsb[1, k:k+1, gdat.indxxpos]])
                        ytemp = np.concatenate([ytemp, starsb[0, k:k+1, gdat.indxypos], starsb[1, k:k+1, gdat.indxypos]])
        
                        neighi = neighbours(xtemp, ytemp, self.scalspmrposi_g, self.n)
                        neighj = neighbours(xtemp, ytemp, self.scalspmrposi_g, self.n+1)
                        if neighi > 0 and neighj > 0:
                            weigoverpairs[k] = 1./neighi + 1./neighj
                        else:
                            weigoverpairs[k] = 0.
                    weigoverpairs *= 0.5 * np.exp(-dr2/(2.*self.scalspmrposi_g*self.scalspmrposi_g))
                    weigoverpairs[weigoverpairs == 0] = 1
                # merge
                elif not boolsplt and idx_reg.size > 1: # need two things to merge!
                    numbspmr = min(numbspmr, idx_reg.size/2, gdat.maxmnumbgalx-self.ng)
                    idx_kill = np.empty((numbspmr, 2), dtype=np.int)
                    choosable = np.zeros(gdat.maxmnumbstar, dtype=np.bool)
                    choosable[idx_reg] = True
                    nchoosable = float(idx_reg.size)
                    invpairs = np.empty(numbspmr)
        
                    for k in xrange(numbspmr):
                        idx_kill[k,0] = np.random.choice(gdat.maxmnumbstar, p=choosable/nchoosable)
                        invpairs[k], idx_kill[k,1] = neighbours(self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.scalspmrposi_g, idx_kill[k,0], generate=True)
                        # prevent sources from being involved in multiple proposals
                        if not choosable[idx_kill[k,1]]:
                            idx_kill[k,1] = -1
                        if idx_kill[k,1] != -1:
                            invpairs[k] = 1./invpairs[k]
                            invpairs[k] += 1./neighbours(self.stars[gdat.indxxpos, 0:self.n], self.stars[gdat.indxypos, 0:self.n], self.scalspmrposi_g, idx_kill[k,1])
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
                    weigoverpairs = np.exp(-dr2/(2.*self.scalspmrposi_g*self.scalspmrposi_g)) * invpairs
                    weigoverpairs[weigoverpairs == 0] = 1
                    fluxspmrtotl = np.sum(fk, axis=1)
                    fluxnormminm = fluxspmrtotl / gdat.fittminmflux
                    frac = fk[:,0] / fluxspmrtotl
                    f1Mf = frac * (1. - frac)
                    galaxiesb = np.empty((6, numbspmr))
                    galaxiesb[gdat.indxxpos,:] = frac*starsk[gdat.indxxpos,:,0] + (1-frac)*starsk[gdat.indxxpos,:,1]
                    galaxiesb[gdat.indxypos,:] = frac*starsk[gdat.indxypos,:,0] + (1-frac)*starsk[gdat.indxypos,:,1]
                    galaxiesb[gdat.indxflux,:] = fluxspmrtotl
                    u = np.random.uniform(low=3e-4, high=1., size=numbspmr).astype(np.float32) #3e-4 for numerics
                    theta = np.arccos(u).astype(np.float32)
                    galaxiesb[gdat.indxxxmo,:] = f1Mf*(dx*dx+u*u*dy*dy)
                    galaxiesb[gdat.indxxymo,:] = f1Mf*(1-u*u)*dx*dy
                    galaxiesb[gdat.indxyymo,:] = f1Mf*(dy*dy+u*u*dx*dx)
                    # this move proposes a splittable galaxy
                    numbbrgt += 1
                    if goodmove:
                        proposal.add_death_stars(idx_kill, starsk)
                        proposal.add_birth_galaxies(galaxiesb)
                if goodmove:
                    factor = 2*np.log(gdat.fittfluxdistslop-1) - np.log(gdat.fittfluxdistslopgalx-1) + 2*(gdat.fittfluxdistslop-1)*np.log(gdat.fittminmflux) - (gdat.fittfluxdistslopgalx-1)*np.log(gdat.fittminmfluxgalx) - \
                        gdat.fittfluxdistslop*np.log(f1Mf) - (2*gdat.fittfluxdistslop - gdat.fittfluxdistslopgalx)*np.log(fluxspmrtotl) - np.log(gdat.sizeimag[0]*gdat.sizeimag[1]) + np.log(1. - 2./fluxnormminm) - \
                        np.log(2*np.pi*self.scalspmrposi_g*self.scalspmrposi_g) + np.log(numbbrgt/(self.ng+1.-boolsplt)) + np.log((self.n-1+2*boolsplt)*weigoverpairs) + \
                        np.log(fluxspmrtotl) - np.log(4.) - 2*np.log(dr2) - 3*np.log(f1Mf) - np.log(np.cos(theta)) - 3*np.log(np.sin(theta))
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
                fluxspmrtotl = 0
                low_n = 0
                idx_bright = idx_reg_g.take(np.flatnonzero(self.galaxies[gdat.indxflux, :].take(idx_reg_g) > gdat.fittminmflux + gdat.fittminmfluxgalx)) # in region and bright enough to make s+g
                numbbrgt = idx_bright.size
        
                numbspmr = (self.nregx * self.nregy) / 4
                goodmove = False
                proposal = Proposal()
                # split off star
                if boolsplt and self.ng > 0 and self.n < gdat.maxmnumbstar and numbbrgt > 0: # need something to split, but don't exceed maxmnumbstar
                    numbspmr = min(numbspmr, numbbrgt, gdat.maxmnumbstar-self.n) # need bright source AND room for split off star
                    dx = (np.random.normal(size=numbspmr)*self.scalspmrposi_g).astype(np.float32)
                    dy = (np.random.normal(size=numbspmr)*self.scalspmrposi_g).astype(np.float32)
                    idx_move_g = np.random.choice(idx_bright, size=numbspmr, replace=False)
                    galaxies0 = self.galaxies.take(idx_move_g, axis=1)
                    x0g, y0g, f0g, xx0g, xy0g, yy0g = galaxies0
                    frac = (gdat.fittminmfluxgalx/f0g + np.random.uniform(size=numbspmr)*(1. - (gdat.fittminmfluxgalx + gdat.fittminmflux)/f0g)).astype(np.float32)
                    galaxiesp = np.empty_like(galaxies0)
                    galaxiesp[gdat.indxxpos,:] = x0g + ((1-frac)*dx)
                    galaxiesp[gdat.indxypos,:] = y0g + ((1-frac)*dy)
                    galaxiesp[gdat.indxflux,:] = f0g * frac
                    pxxg = (xx0g - frac*(1-frac)*dx*dx)/frac
                    pxyg = (xy0g - frac*(1-frac)*dx*dy)/frac
                    pyyg = (yy0g - frac*(1-frac)*dy*dy)/frac
                    galaxiesp[gdat.indxxxmo,:] = pxxg
                    galaxiesp[gdat.indxxymo,:] = pxyg
                    galaxiesp[gdat.indxyymo,:] = pyyg
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
                    fluxspmrtotl = f0g
                    invpairs = np.zeros(numbspmr)
                    for k in xrange(numbspmr):
                        xtemp = self.stars[gdat.indxxpos, 0:self.n].copy()
                        ytemp = self.stars[gdat.indxypos, 0:self.n].copy()
                        xtemp = np.concatenate([xtemp, galaxiesp[gdat.indxxpos, k:k+1], starsb[gdat.indxxpos, k:k+1]])
                        ytemp = np.concatenate([ytemp, galaxiesp[gdat.indxypos, k:k+1], starsb[gdat.indxypos, k:k+1]])
        
                        invpairs[k] =  1./neighbours(xtemp, ytemp, self.scalspmrposi_g, self.n)
                # merge galaxy + star into galaxy
                elif not boolsplt and idx_reg_g.size > 1: # need two things to merge!
                    numbspmr = min(numbspmr, idx_reg_g.size)
                    idx_move_g = np.random.choice(idx_reg_g, size=numbspmr, replace=False) # choose galaxies and then see if they have neighbours
                    idx_kill = np.empty(numbspmr, dtype=np.int)
                    choosable = np.full(gdat.maxmnumbstar, True, dtype=np.bool)
                    nchoosable = float(gdat.maxmnumbstar)
                    invpairs = np.empty(numbspmr)
        
                    for k in xrange(numbspmr):
                        l = idx_move_g[k]
                        invpairs[k], idx_kill[k] = neighbours(np.concatenate([self.stars[gdat.indxxpos, 0:self.n], self.galaxies[gdat.indxxpos, l:l+1]]), \
                                np.concatenate([self.stars[gdat.indxypos, 0:self.n], self.galaxies[gdat.indxypos, l:l+1]]), self.scalspmrposi_g, self.n, generate=True)
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
                    if gdat.priotype == 'prio':
                        indxfluxtemp = gdat.indxflux[0, 0]
                    else:
                        indxfluxtemp = gdat.indxflux
                    fluxspmrtotl = galaxies0[indxfluxtemp, :] + starsk[indxfluxtemp, :]
                    frac = galaxies0[indxfluxtemp, :] / fluxspmrtotl
                    if gdat.priotype != 'prio':
                        frac = np.mean(frac, axis=(0, 1))
                    dx = galaxies0[gdat.indxxpos,:] - starsk[gdat.indxxpos,:]
                    dy = galaxies0[gdat.indxypos,:] - starsk[gdat.indxypos,:]
                    pxxg = frac*galaxies0[gdat.indxxxmo,:] + frac*(1-frac)*dx*dx
                    pxyg = frac*galaxies0[gdat.indxxymo,:] + frac*(1-frac)*dx*dy
                    pyyg = frac*galaxies0[gdat.indxyymo,:] + frac*(1-frac)*dy*dy
                    inbounds = (pxxg > 0) * (pyyg > 0) * (pxxg*pyyg > pxyg*pxyg) # make sure ellipse is legal TODO min galaxy radius
                    idx_move_g = idx_move_g.compress(inbounds)
                    galaxies0 = galaxies0.compress(inbounds, axis=1)
                    idx_kill = idx_kill.compress(inbounds)
                    invpairs = invpairs.compress(inbounds)
                    starsk = starsk.compress(inbounds, axis=1)
                    fluxspmrtotl = fluxspmrtotl.compress(inbounds)
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
                        galaxiesp[gdat.indxxxmo,:] = pxxg
                        galaxiesp[gdat.indxxymo,:] = pxyg
                        galaxiesp[gdat.indxyymo,:] = pyyg
                        proposal.add_move_galaxies(idx_move_g, galaxies0, galaxiesp)
                        proposal.add_death_stars(idx_kill, starsk)
                    # this proposal makes a galaxy that is bright enough to split
                    numbbrgt = numbbrgt + 1
                if goodmove:
                    factor = np.log(gdat.fittfluxdistslop-1) - (gdat.fittfluxdistslop-1)*np.log(fluxspmrtotl/gdat.fittminmflux) - gdat.fittfluxdistslopgalx*np.log(frac) - gdat.fittfluxdistslop*np.log(1-frac) + \
                            np.log(2*np.pi*self.scalspmrposi_g*self.scalspmrposi_g) - np.log(gdat.sizeimag[0]*gdat.sizeimag[1]) + np.log(1. - (gdat.fittminmflux+gdat.fittminmfluxgalx)/fluxspmrtotl) + \
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
                fluxspmrtotl = 0
                low_n = 0
                idx_bright = idx_reg.take(np.flatnonzero(self.galaxies[gdat.indxflux, :].take(idx_reg) > 2*gdat.fittminmfluxgalx)) # in region!
                numbbrgt = idx_bright.size
        
                numbspmr = (self.nregx * self.nregy) / 4
                goodmove = False
                proposal = Proposal()
                # split
                if boolsplt and self.ng > 0 and self.ng < gdat.maxmnumbgalx and numbbrgt > 0: # need something to split, but don't exceed maxmnumbstar
                    numbspmr = min(numbspmr, numbbrgt, gdat.maxmnumbgalx-self.ng) # need bright source AND room for split source
                    dx = (np.random.normal(size=numbspmr)*self.scalspmrposi_g).astype(np.float32)
                    dy = (np.random.normal(size=numbspmr)*self.scalspmrposi_g).astype(np.float32)
                    idx_move_g = np.random.choice(idx_bright, size=numbspmr, replace=False)
                    galaxies0 = self.galaxies.take(idx_move_g, axis=1)
                    fluxnormminm = galaxies0[gdat.indxflux,:] / gdat.fittminmfluxgalx
                    frac = (1./fluxnormminm + np.random.uniform(size=numbspmr)*(1. - 2./fluxnormminm)).astype(np.float32)
                    frac_xx = np.random.uniform(size=numbspmr).astype(np.float32)
                    frac_xy = np.random.uniform(size=numbspmr).astype(np.float32)
                    frac_yy = np.random.uniform(size=numbspmr).astype(np.float32)
                    xx_p = galaxies0[gdat.indxxxmo,:] - frac*(1-frac)*dx*dx# moments of just galaxy pair
                    xy_p = galaxies0[gdat.indxxymo,:] - frac*(1-frac)*dx*dy
                    yy_p = galaxies0[gdat.indxyymo,:] - frac*(1-frac)*dy*dy
                    galaxiesp = np.empty_like(galaxies0)
                    galaxiesp[gdat.indxxpos,:] = galaxies0[gdat.indxxpos,:] + ((1-frac)*dx)
                    galaxiesp[gdat.indxypos,:] = galaxies0[gdat.indxypos,:] + ((1-frac)*dy)
                    galaxiesp[gdat.indxflux,:] = galaxies0[gdat.indxflux,:] * frac
                    galaxiesp[gdat.indxxxmo,:] = xx_p * frac_xx / frac
                    galaxiesp[gdat.indxxymo,:] = xy_p * frac_xy / frac
                    galaxiesp[gdat.indxyymo,:] = yy_p * frac_yy / frac
                    galaxiesb = np.empty_like(galaxies0)
                    galaxiesb[gdat.indxxpos,:] = galaxies0[gdat.indxxpos,:] - frac*dx
                    galaxiesb[gdat.indxypos,:] = galaxies0[gdat.indxypos,:] - frac*dy
                    galaxiesb[gdat.indxflux,:] = galaxies0[gdat.indxflux,:] * (1-frac)
                    galaxiesb[gdat.indxxxmo,:] = xx_p * (1-frac_xx) / (1-frac) # FIXME is this right?
                    galaxiesb[gdat.indxxymo,:] = xy_p * (1-frac_xy) / (1-frac)
                    galaxiesb[gdat.indxyymo,:] = yy_p * (1-frac_yy) / (1-frac)
                    # don't want to think about how to bounce split-merge
                    # don't need to check if above fmin, because of how frac is decided
                    pxxg, pxyg, pyyg = galaxiesp[[gdat.indxxxmo,gdat.indxxymo,gdat.indxyymo],:]
                    bxxg, bxyg, byyg = galaxiesb[[gdat.indxxxmo,gdat.indxxymo,gdat.indxyymo],:]
                    inbounds = self.in_bounds(galaxiesp) * \
                               self.in_bounds(galaxiesb) * \
                               (pxxg > 0) * (pyyg > 0) * (pxxg*pyyg > pxyg*pxyg) * \
                               (bxxg > 0) * (byyg > 0) * (bxxg*byyg > bxyg*bxyg) # TODO min galaxy rad
                    galaxies0 = galaxies0.compress(inbounds, axis=1)
                    galaxiesp = galaxiesp.compress(inbounds, axis=1)
                    galaxiesb = galaxiesb.compress(inbounds, axis=1)
                    idx_move_g = idx_move_g.compress(inbounds)
                    fluxnormminm = fluxnormminm.compress(inbounds)
                    frac = frac.compress(inbounds)
                    xx_p = xx_p.compress(inbounds)
                    xy_p = xy_p.compress(inbounds)
                    yy_p = yy_p.compress(inbounds)
                    numbspmr = idx_move_g.size
                    goodmove = numbspmr > 0
        
                    # need to calculate factor
                    fluxspmrtotl = galaxies0[gdat.indxflux,:]
                    invpairs = np.empty(numbspmr)
                    for k in xrange(numbspmr):
                        xtemp = self.galaxies[gdat.indxxpos, 0:self.ng].copy()
                        ytemp = self.galaxies[gdat.indxypos, 0:self.ng].copy()
                        xtemp[idx_move_g[k]] = galaxiesp[gdat.indxxpos, k]
                        ytemp[idx_move_g[k]] = galaxiesp[gdat.indxypos, k]
                        xtemp = np.concatenate([xtemp, galaxiesb[gdat.indxxpos, k:k+1]])
                        ytemp = np.concatenate([ytemp, galaxiesb[gdat.indxypos, k:k+1]])
        
                        invpairs[k] =  1./neighbours(xtemp, ytemp, self.scalspmrposi_g, idx_move_g[k])
                        invpairs[k] += 1./neighbours(xtemp, ytemp, self.scalspmrposi_g, self.ng)
                    invpairs *= 0.5
                # merge
                elif not boolsplt and idx_reg.size > 1: # need two things to merge!
                    numbspmr = min(numbspmr, idx_reg.size/2)
                    idx_move_g = np.empty(numbspmr, dtype=np.int)
                    idx_kill_g = np.empty(numbspmr, dtype=np.int)
                    choosable = np.zeros(gdat.maxmnumbgalx, dtype=np.bool)
                    choosable[idx_reg] = True
                    nchoosable = float(idx_reg.size)
                    invpairs = np.empty(numbspmr)
        
                    for k in xrange(numbspmr):
                        idx_move_g[k] = np.random.choice(gdat.maxmnumbgalx, p=choosable/nchoosable)
                        invpairs[k], idx_kill_g[k] = neighbours(self.galaxies[gdat.indxxpos, 0:self.ng], \
                                                                self.galaxies[gdat.indxypos, 0:self.ng], self.scalspmrposi_g, idx_move_g[k], generate=True)
                        if invpairs[k] > 0:
                            invpairs[k] = 1./invpairs[k]
                        # prevent sources from being involved in multiple proposals
                        if not choosable[idx_kill_g[k]]:
                            idx_kill_g[k] = -1
                        if idx_kill_g[k] != -1:
                            invpairs[k] += 1./neighbours(self.galaxies[gdat.indxxpos, 0:self.ng], self.galaxies[gdat.indxypos, 0:self.ng], self.scalspmrposi_g, idx_kill_g[k])
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
                    fluxspmrtotl = galaxies0[gdat.indxflux,:] + galaxiesk[gdat.indxflux,:]
                    fluxnormminm = fluxspmrtotl / gdat.fittminmfluxgalx
                    frac = galaxies0[gdat.indxflux,:] / fluxspmrtotl
        
                    galaxiesp = np.empty_like(galaxies0)
                    galaxiesp[gdat.indxxpos,:] = frac*galaxies0[gdat.indxxpos,:] + (1-frac)*galaxiesk[gdat.indxxpos,:]
                    galaxiesp[gdat.indxypos,:] = frac*galaxies0[gdat.indxypos,:] + (1-frac)*galaxiesk[gdat.indxypos,:]
                    galaxiesp[gdat.indxflux,:] = galaxies0[gdat.indxflux,:] + galaxiesk[gdat.indxflux,:]
                    dx = galaxies0[gdat.indxxpos,:] - galaxiesk[gdat.indxxpos,:]
                    dy = galaxies0[gdat.indxypos,:] - galaxiesk[gdat.indxypos,:]
                    xx_p = frac*galaxies0[gdat.indxxxmo,:] + (1-frac)*galaxiesk[gdat.indxxxmo,:]
                    xy_p = frac*galaxies0[gdat.indxxymo,:] + (1-frac)*galaxiesk[gdat.indxxymo,:]
                    yy_p = frac*galaxies0[gdat.indxyymo,:] + (1-frac)*galaxiesk[gdat.indxyymo,:]
                    galaxiesp[gdat.indxxxmo,:] = xx_p + frac*(1-frac)*dx*dx
                    galaxiesp[gdat.indxxymo,:] = xy_p + frac*(1-frac)*dx*dy
                    galaxiesp[gdat.indxyymo,:] = yy_p + frac*(1-frac)*dy*dy
        
                    pxxg, pxyg, pyyg = galaxiesp[[gdat.indxxxmo,gdat.indxxymo,gdat.indxyymo],:]
                    inbounds = ((pxxg > 0) * (pyyg > 0) * (pxxg*pyyg > pxyg*pxyg)) # ellipse legal TODO minimum radius
                    galaxies0 = galaxies0.compress(inbounds, axis=1)
                    galaxiesk = galaxiesk.compress(inbounds, axis=1)
                    galaxiesp = galaxiesp.compress(inbounds, axis=1)
                    idx_move_g = idx_move_g.compress(inbounds)
                    idx_kill_g = idx_kill_g.compress(inbounds)
                    invpairs = invpairs.compress(inbounds)
                    fluxspmrtotl = fluxspmrtotl.compress(inbounds)
                    fluxnormminm = fluxnormminm.compress(inbounds)
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
                    numbbrgt = numbbrgt - (galaxies0[gdat.indxflux,:] > 2*gdat.fittminmfluxgalx) - (galaxiesk[gdat.indxflux,:] > 2*gdat.fittminmfluxgalx) + \
                                                                                                   (galaxiesp[gdat.indxflux,:] > 2*gdat.fittminmfluxgalx)
                if goodmove:
                    factor = np.log(gdat.fittfluxdistslopgalx-1) + (gdat.fittfluxdistslopgalx-1)*np.log(gdat.fittminmflux) \
                        - gdat.fittfluxdistslopgalx * np.log(frac*(1-frac)*fluxspmrtotl) + \
                        np.log(2 * np.pi*self.scalspmrposi_g*self.scalspmrposi_g) - np.log(gdat.sizeimag[0]*gdat.sizeimag[1]) + \
                        np.log(1. - 2./fluxnormminm) + np.log(numbbrgt) + np.log(invpairs) + \
                        np.log(fluxspmrtotl) + np.log(xx_p) + np.log(np.abs(xy_p)) + np.log(yy_p) - 3*np.log(frac) - 3*np.log(1-frac) # last line is Jacobian
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
        
        # construct model for each temperature
        listobjtmodl = [Model() for k in xrange(ntemps)]
        
        # write the chain
        ## h5 file path
        pathh5py = gdat.pathdatartag + gdat.rtag + '_chan.h5'
        ## numpy object file path
        pathnump = gdat.pathdatartag + gdat.rtag + '_chan.npz'
        
        if gdat.boolplot:
            plt.figure(figsize=(21, 7))
        
        if gdat.strgstat != None:
            gdat.catlmlik = {}
            gdat.catlmlik['numb'] = 0
            for strgfeat in gdat.liststrgfeatstar:
                gdat.catlmlik[strgfeat] = np.zeros(gdat.maxmnumbstar)
        
        gdat.indxswep = np.arange(gdat.numbswep)
        gdat.indxswlp = np.arange(gdat.numbswlp)
        chan = {}
        chan['numb'] = np.zeros(gdat.numbsamp, dtype=np.int32)
        if gdat.strgmodl == 'galx':
            chan['numbgalx'] = np.zeros(gdat.numbsamp, dtype=np.int32)
        chan['chi2'] = np.zeros(gdat.numbsamp)
        chan['lposterm'] = np.zeros((gdat.numbswlp, gdat.numbregixaxi, gdat.numbregiyaxi))
        for strgfeat in gdat.liststrgfeatstar:
            if strgfeat == 'flux':
                chan[strgfeat] = np.zeros((gdat.numbsamp, gdat.numbener, gdat.numbtime, gdat.maxmnumbstar))
            else:
                chan[strgfeat] = np.zeros((gdat.numbsamp, gdat.maxmnumbstar))
        
        for strgfeat in gdat.liststrgfeatgalx:
            if strgfeat == 'numbgalx':
                chan[strgfeat] = np.zeros((gdat.numbsamp, gdat.numbener, gdat.numbtime, gdat.maxmnumbgalx))
            elif strgfeat == 'flux':
                chan[strgfeat + 'galx'] = np.zeros((gdat.numbsamp, gdat.numbener, gdat.numbtime, gdat.maxmnumbgalx))
            else:
                chan[strgfeat + 'galx'] = np.zeros((gdat.numbsamp, gdat.maxmnumbgalx))
        
        chan['proptype'] = np.zeros((gdat.numbswlp))
        chan['booloutb'] =  np.zeros((gdat.numbswlp))
        chan['stdvfluxprop'] =  np.zeros((gdat.numbswlp))
        chan['stdvposiprop'] =  np.zeros((gdat.numbswlp))
        chan['deltllik'] =  np.zeros((gdat.numbregiyaxi, gdat.numbregixaxi, gdat.numbswlp))
        chan['accp'] =  np.zeros((gdat.numbregiyaxi, gdat.numbregixaxi, gdat.numbswlp))
        chan['dt01'] = np.zeros((gdat.numbswlp))
        chan['dt02'] = np.zeros((gdat.numbswlp))
        chan['dt03'] = np.zeros((gdat.numbswlp))
                
        mlik = -1e100
        for j in gdat.indxswep:
            listchi2totl = np.zeros(ntemps)
        
            #temptemp = max(50 - 0.1*j, 1)
            temptemp = 1.
            for k in xrange(ntemps):
                _, _, listchi2totl[k] = listobjtmodl[k].run_sampler(gdat, chan, temptemp, j)
        
            lliktotl = -0.5 * listchi2totl[0]

            for k in xrange(ntemps-1, 0, -1):
                logfac = (listchi2totl[k-1] - listchi2totl[k]) * (1./temps[k-1] - 1./temps[k]) / 2.
                if np.log(np.random.uniform()) < logfac:
                    print 'swapped', k-1, k
                    listobjtmodl[k-1], listobjtmodl[k] = listobjtmodl[k], listobjtmodl[k-1]
            
            if gdat.strgstat != None and j % gdat.factthinplot == 0:
                print 'Updating the maximum likelihood catalog...'
                # temp -- assumes single temperature
                # save the sample and record the likelihood
                if lliktotl > mlik:
                    gdat.catlmlik['numb'] = listobjtmodl[0].n
                    for strgfeat in gdat.liststrgfeatstar:
                        indx = getattr(gdat, 'indx' + strgfeat)
                        gdat.catlmlik[strgfeat] = listobjtmodl[0].stars[indx, :]
                    mlik = lliktotl
                    writ_catl(gdat, gdat.catlmlik, gdat.pathstat)
                #print 'lliktotl'
                #print lliktotl
                #print 'mlik'
                #print mlik

            if gdat.boolsaveswep[j]:
                chan['numb'][j-gdat.numbswepburn] = listobjtmodl[0].n
                for strgfeat in gdat.liststrgfeatstar:
                    indx = getattr(gdat, 'indx' + strgfeat)
                    chan[strgfeat][j-gdat.numbswepburn, :] = listobjtmodl[0].stars[indx, :]
                if gdat.strgmodl == 'galx':
                    chan['numbgalx'][j-gdat.numbswepburn] = listobjtmodl[0].ng
                    for strgfeat in gdat.liststrgfeatgalx:
                        indx = getattr(gdat, 'indx' + strgfeat)
                        if strgfeat != 'numbgalx':
                            strgfeat += 'galx'
                        chan[strgfeat][j-gdat.numbswepburn, :] = listobjtmodl[0].galaxies[indx, :]
                chan['chi2'][j-gdat.numbswepburn] = np.sum(listchi2totl[0])
   
        writ_catl(gdat, chan, pathh5py)
        
        path = gdat.pathdatartag + 'gdat.p'
        filepick = open(path, 'wb')
        print 'Writing to %s...' % path
        
        #for attr, valu in gdat.__dict__.iteritems():
        #    print 'attr'
        #    print attr
        #    print 'valu'
        #    print valu
        #    gdattemp = gdatstrt()
        #    gdattemp.temp = valu
        #    filepick = open(path, 'wb')
        #    cPickle.dump(gdattemp, filepick, protocol=cPickle.HIGHEST_PROTOCOL)
        #    filepick.close()
        cPickle.dump(gdat, filepick, protocol=cPickle.HIGHEST_PROTOCOL)
        filepick.close()
    
    else:
        path = gdat.pathdata + gdat.strgproc + '/gdat.p'
        filepick = open(path, 'rb')
        print 'Reading %s...' % path
        gdat = cPickle.load(filepick)
        filepick.close()
        
        print 'gdat.pathdatartag'
        print gdat.pathdatartag
        pathchan = gdat.pathdatartag + gdat.rtag + '_chan.h5'
        chan = read_catl(gdat, pathchan)
        
    # calculate the condensed catalog
    gdat.catlcond = retr_catlcond(gdat.rtag, gdat.pathdata, pathdatartag=gdat.pathdatartag)
    if gdat.strgmodl == 'galx':
        gdat.catlcond = retr_catlcond(gdat.rtag, gdat.pathdata, pathdatartag=gdat.pathdatartag, boolgalx=True)
    
    if gdat.catlcond is not None:
        gdat.numbsourcond = gdat.catlcond.shape[1]
        print 'gdat.numbsourcond'
        print gdat.numbsourcond
        gdat.indxsourcond = np.arange(gdat.numbsourcond)
    
    plot_pcat(gdat)

    if gdat.catlcond is not None:
        if gdat.numbtime > 1:
            print 'Plotting light curves...'
            plot_lcur(gdat)
            plot_lcur(gdat, boolsimp=True)

    if gdat.boolplot:
        
        # posterior plots
        # plot the condensed catalog
        if gdat.boolplotsave:
           
            listtemp = [ \
            # temp
            #            ['lposterm', '$\log P_*$'], \
                        ['booloutb', '$b_o$'], \
                        ['stdvfluxprop', '$\sigma_{p,f}$'], \
                        ['stdvposiprop', '$\sigma_{p,r}$'], \
                       ]
            #for name, labl in listtemp:
            #    plot_varbsamp(gdat, chan[name], labl, name, xaxitype='swlp')
            #listtemp = [ \
            #            ['numb', '$N_{star}$'], \
            #            ['chi2', '$\chi^2$'], \
            #           ]
            #for name, labl in listtemp:
            #    plot_varbsamp(gdat, chan[name], labl, name)
           

            #for u in range(gdat.numbregixaxi):
            #    for v in range(gdat.numbregiyaxi):
            #        retr_meanmove(chan['accp'][v, u, :])
            #        plot_varbsamp(gdat, retr_meanmove(chan['accp'][v, u, :]), '$A_{%d,%d}$' % (v, u), 'accp', xaxitype='swlp')
            #        
            #        for k in range(3):
            #            indxprop = np.where(chan['proptype'] == k)[0]
            #            temp = np.zeros_like(chan['accp'][v, u, :])
            #            print 'chan[accp][v, u, :]'
            #            summgene(chan['accp'][v, u, :])
            #            print 'chan[accp][v, u, indxprop]'
            #            summgene(chan['accp'][v, u, indxprop])
            #            #indx = temp[0]
            #            temp[indxprop] = retr_meanmove(chan['accp'][v, u, indxprop])
            #            plot_varbsamp(gdat, temp, '$A_{%d,%d,%d}$' % (v, u, k), 'accp%d%d%02d' % (v, u, k), xaxitype='swlp')
            #            deltlliktemp = np.zeros_like(chan['deltllik'][v, u, :])
            #            deltlliktemp[indxprop] = chan['deltllik'][v, u, indxprop]
            #            plot_varbsamp(gdat, deltlliktemp, '$\Delta\ln P(D|M)_{%d,%d,%d}$' % (v, u, k), 'deltllik%d%d%d' % (v, u, k), xaxitype='swlp')
            
            print 'Plotting the condensed catalog...'
            
            numbstarplot = min(10, gdat.maxmnumbstar)
            indxstarplot = np.arange(numbstarplot)

            for k in indxstarplot:
                
                # horizontal and vertical positions
                plot_varbsamp(gdat, chan['xpos'][:, k], '$x_{%d}$' % k, 'xpos%04d' % k)
                plot_varbsamp(gdat, chan['ypos'][:, k], '$y_{%d}$' % k, 'ypos%04d' % k)
                
                # fluxes
                for i in gdat.indxener:
                    for t in gdat.indxtime:
                        plot_varbsamp(gdat, chan['flux'][:, i, t, k], '$f_{%d,%d,%d}$' % (i, t, k), 'flux%04d%04d%04d' % (i, t, k))
            
            if gdat.catlcond is not None:
                # flux distribution
                plot_fluxhist(gdat, gdat.catlcond[gdat.indxflux, :, 0], 0, 0)
                
                for i in gdat.indxener:
                    for t in gdat.indxtime:
                        # magnitude distribution
                        plot_fluxhist(gdat, gdat.catlcond[gdat.indxflux, :, 0], i, t, plotmagt=True)
                
                        # spatial distribution
                        figr, axis = plt.subplots()
                        axis.imshow(gdat.cntpdata[i, :, :, t], origin='lower', interpolation='nearest', cmap='Greys_r', vmin=gdat.minmcntpdata, vmax=gdat.maxmcntpdata)
                        numbsampplot = min(gdat.numbsamp, 10)
                        indxsampplot = np.random.choice(gdat.indxsamp, size=numbsampplot, replace=False) 
                        for k in range(numbsampplot):
                            numb = chan['numb'][indxsampplot[k]]
                            supr_catl(gdat, axis, i, t, chan['xpos'][k, :numb], chan['ypos'][k, :numb], chan['flux'][k, :, :, :numb])
                        # temp
                        supr_catl(gdat, axis, i, t, gdat.catlcond[gdat.indxxpos, :, 0], gdat.catlcond[gdat.indxypos, :, 0], gdat.catlcond[gdat.indxflux[i, t], :, 0], boolcond=True)
                        plt.tight_layout()
                        plt.savefig(gdat.pathdatartag + '%s_condcatl%04d%04d.' % (gdat.rtag, i, t) + gdat.strgplotfile)
                        plt.close()
        
    if gdat.boolplotsave:
        print 'Making animations of frame plots...'
        
        for i in gdat.indxener:
            for t in gdat.indxtime:
                
                if not gdat.boolframplotenti and (i != 0 or t != 0):
                    break
                                
                cmnd = 'convert -delay 20 -density 200x200 %s/%s_cntpdata%04d%04d_fram*.%s %s/%s_cntpdata%04d%04d.gif' % (gdat.pathdatartag, gdat.rtag, i, t, gdat.strgplotfile, gdat.pathdatartag, gdat.rtag, i, t)
                print cmnd
                os.system(cmnd)
                cmnd = 'convert -delay 20 -density 200x200 %s/%s_cntpresi%04d%04d_fram*.%s %s/%s_cntpresi%04d%04d.gif' % (gdat.pathdatartag, gdat.rtag, i, t, gdat.strgplotfile, gdat.pathdatartag, gdat.rtag, i, t)
                print cmnd
                os.system(cmnd)
        print 'Done.'
    

def retr_meanmove(arry, numbkern=10):

    arrymean = np.empty_like(arry)
    numb = arry.size
    indx = np.arange(numb)
    for k in indx:
        indxloww = max(0, k - numbkern)
        arrymean[k] = np.mean(arry[indxloww:k+1])

    return arrymean


def samp_prio_momegalx(truermin_g, ngalx, slope=np.float32(4)):
    
    rg = truermin_g*np.exp(np.random.exponential(scale=1./(slope-1.),size=ngalx)).astype(np.float32)
    ug = np.random.uniform(low=3e-4, high=1., size=ngalx).astype(np.float32) #3e-4 for numerics
    thetag = np.arccos(ug).astype(np.float32)
    phig = (np.random.uniform(size=ngalx)*np.pi - np.pi/2.).astype(np.float32)
    
    return to_moments(rg, thetag, phig)
            

def retr_catlseed(rtag, pathdata, pathdatartag=None, boolgalx=False):
    
    strgtimestmp = rtag[:15]
    
    pathlion, pathdata = retr_path()
    
    if pathdatartag is None:
        pathdatartag = pathdata + rtag + '/'
    path = pathdatartag + 'gdat.p'
    filepick = open(path, 'rb')
    print 'Reading %s...' % path
    gdat = cPickle.load(filepick)
    filepick.close()

    os.system('mkdir -p %s' % gdat.pathdatartag)

    # number of samples used in the seed catalog determination
    numbsampseed = min(10, gdat.numbsamp)
    pathchan = gdat.pathdatartag + rtag + '_chan.h5'
    chan = read_catl(gdat, pathchan)
   
    # get the initial numbsampseed samples to determine the seed catalog 
    for strgfeat in gdat.liststrgfeatstar:
        chan[strgfeat] = chan[strgfeat][:numbsampseed, :]
    
    PCi, junk = np.mgrid[:numbsampseed, :gdat.maxmnumbstar]
    
    # temp - uses only first energy, first time
    mask = chan['flux'][:, 0, 0, :] > 0
    if np.sum(mask) == 0:
        return
    PCc_all = np.zeros((np.sum(mask), 2))
    
    if gdat.strgmodl == 'galx':
        strgtemp = gdat.strgmodl
    else:   
        strgtemp = ''

    PCc_all[:, 0] = chan['xpos' + strgtemp][mask].flatten()
    PCc_all[:, 1] = chan['ypos' + strgtemp][mask].flatten()
    PCi = PCi[mask].flatten()
    
    print 'Constructing a KD tree...'
    kdtree = scipy.spatial.KDTree(PCc_all)
    matches = kdtree.query_ball_tree(kdtree, 0.75)
    
    print 'Constructing the network...'
    G = networkx.Graph()
    G.add_nodes_from(xrange(0, PCc_all.shape[0]))
    # for each source, add edges to all other sources from other catalogues within match radius
    for i in xrange(PCc_all.shape[0]):
        for j in matches[i]:
            if PCi[i] != PCi[j]:
                G.add_edge(i, j)
    
    print 'Iterating over sources...'

    for i in xrange(PCc_all.shape[0]):
        # sort matches so they are in order of catalogues they belong to
        matches[i].sort()
        bincount = np.bincount(PCi[matches[i]]).astype(np.int) # how many matches are from each catalogue
        ending = np.cumsum(bincount).astype(np.int) # ending index for each catalogue
        starting = np.zeros(bincount.size).astype(np.int)
        starting[1:bincount.size] = ending[0:bincount.size-1] # starting index for each catalogue
        # for each other catalogue
        for j in xrange(bincount.size):
            if j == PCi[i]: # do not match to same catalogue
                continue
            if bincount[j] == 0: # no match to catalogue j
                continue
            if bincount[j] == 1: # exactly one match to catalogue j
                continue
            if bincount[j] > 1:
                # find closest match in catalogue j
                dist2 = 0.75**2
                l = -1
                for k in xrange(starting[j], ending[j]):
                    m = matches[i][k]
                    newdist2 = np.sum((PCc_all[i,:] - PCc_all[m,:])**2)
                    if newdist2 < dist2:
                        l = m
                        dist2 = newdist2
                if l == -1:
                    print "didn't find edge even though multiple matches from this catalogue?"
                # only keep edge to closest match
                for k in xrange(starting[j], ending[j]):
                    m = matches[i][k]
                    if m != l:
                        if G.has_edge(i, m):
                            G.remove_edge(i, m)
    
    catlseed = []
    # make seed catalogue by taking graph nodes in order of most edges, updating graph as we go
    while networkx.number_of_nodes(G) > 0:
        deg = networkx.degree(G)
        maxmdegr = 0
        i = 0
        for node in G:
            if deg[node] >= maxmdegr:
                i = node
                maxmdegr = deg[node]

        neighbors = networkx.all_neighbors(G, i)
        catlseed.append([PCc_all[i, 0], PCc_all[i, 1], deg[i]])
        G.remove_node(i)
        G.remove_nodes_from(neighbors)
    
    catlseedtemp = np.array(catlseed)
    
    catlseed = {}
    gdat.numbsourseed = catlseedtemp.shape[0]

    catlseed['numb' + strgtemp] = gdat.numbsourseed
    catlseed['xpos' + strgtemp] = catlseedtemp[:, 0]
    catlseed['ypos' + strgtemp] = catlseedtemp[:, 1]
    catlseed['degr' + strgtemp] = catlseedtemp[:, 2]
    
    if gdat.strgmodl == 'galx':
        strgtemptemp = '_'
    else:
        strgtemptemp = ''
    pathcatlseed = gdat.pathdatartag + rtag + '_catlseed%s%s.h5' % (strgtemptemp, strgtemp)
    writ_catl(gdat, catlseed, pathcatlseed)


def retr_catlcond(rtag, pathdata, pathdatartag=None, boolgalx=False):

    strgtimestmp = rtag[:15]
    
    # paths
    pathlion, pathdata = retr_path()

    path = pathdatartag + 'gdat.p'
    filepick = open(path, 'rb')
    print 'Reading %s...' % path
    gdat = cPickle.load(filepick)
    filepick.close()

    setp(gdat)
    
    if gdat.strgmodl == 'galx':
        strgtemp = gdat.strgmodl
        strgtemptemp = '_'
    else:   
        strgtemp = ''
        strgtemptemp = ''
    
    if pathdatartag is None:
        pathdatartag = pathdata + rtag + '/'
    os.system('mkdir -p %s' % pathdatartag)

    pathcatlcond = pathdatartag + rtag + '_catlcond%s%s.h5' % (strgtemptemp, strgtemp)
    
    # search radius
    radisrch = 0.75
    
    # confidence cut
    minmdegr = 2 
    
    # read the chain
    print 'Reading the chain...'    
    pathchan = pathdatartag + rtag + '_chan.h5'
    chan = read_catl(gdat, pathchan)

    # sort the catalog in decreasing flux
    if gdat.strgmodl == 'galx':
        numbpara = gdat.numbparagalx
    else:
        numbpara = gdat.numbparastar
    catlsort = np.zeros((gdat.numbsamp, gdat.maxmnumbstar, numbpara))
    for b in gdat.indxsamp:
        catl = np.zeros((gdat.maxmnumbstar, gdat.numbparastar))
        catl[:, gdat.indxxpos] = chan['xpos' + strgtemp][b, :]
        catl[:, gdat.indxypos] = chan['ypos' + strgtemp][b, :]
        catl[:, gdat.indxflux.flatten()] = chan['flux' + strgtemp][b, :, :, :].reshape((gdat.numbflux, gdat.maxmnumbstar)).T
        if gdat.strgmodl == 'galx':
            catl[:, gdat.indxxxmo] = chain['xxmogalx'][b,:]
            catl[:, gdat.indxxymo] = chain['xymogalx'][b,:]
            catl[:, gdat.indxyymo] = chain['yymogalx'][b,:]

        # temp - sorts by first energy, first time
        catlsort[b, :, :] = np.flipud(catl[catl[:, gdat.indxflux[0, 0]].argsort(), :])
    print "Stacking catalogs..."
    
    # number of stacked stars
    numbstarstck = np.sum(chan['numb'])
    
    # create stacked array for KD tree creation
    catlstck = np.zeros((numbstarstck, 2))
    cntr = 0
    for b in gdat.indxsamp:
        catlstck[cntr:cntr+chan['numb' + strgtemp][b], 0] = catlsort[b, :chan['numb' + strgtemp][b], gdat.indxxpos]
        catlstck[cntr:cntr+chan['numb' + strgtemp][b], 1] = catlsort[b, :chan['numb' + strgtemp][b], gdat.indxypos]
        cntr += chan['numb' + strgtemp][b]

    retr_catlseed(rtag, gdat.pathdata, pathdatartag=gdat.pathdatartag, boolgalx=boolgalx)
    
    # seed catalog
    ## load the catalog
    pathcatlseed = gdat.pathdatartag + rtag + '_catlseed%s%s.h5' % (strgtemptemp, strgtemp)
    if not os.path.exists(pathcatlseed):
        return None

    catlseed = read_catl(gdat, pathcatlseed)
    
    ## perform confidence cut
    
    indxstar = np.where(catlseed['degr'] >= minmdegr)[0]
    catlseed['xpos'] = catlseed['xpos'][indxstar]
    catlseed['ypos'] = catlseed['ypos'][indxstar]
    catlseed['degr'] = catlseed['degr'][indxstar]
    
    numbsourseed = catlseed['xpos'].size
    
    if numbsourseed == 0:
        return None

    indxsourseed = np.arange(numbsourseed)
    catlseedtemp = np.empty((numbsourseed, 2)) 
    catlseedtemp[:, 0] = catlseed['xpos']
    catlseedtemp[:, 1] = catlseed['ypos']
    catlseed = catlseedtemp
    
    #creates tree, where tree is Pcc_stack
    tree = scipy.spatial.KDTree(catlstck)
    
    # features of the condensed sources
    featcond = np.zeros((gdat.numbsamp, numbsourseed, gdat.numbparastar))
    # prevalences of condensed sources
    prvl = np.zeros(numbsourseed)

    #numpy mask for sources that have been matched
    mask = np.zeros(numbstarstck)
    
    #first, we iterate over all sources in the seed catalog:
    for k in indxsourseed:
    
        #query ball point at seed catalog
        matches = tree.query_ball_point(catlseed[k], radisrch)
    
        #in each catalog, find first instance of match w/ desired source (-- first instance okay b/c every catalog is sorted brightest to faintest)
        ##this for loop should naturally populate original seed catalog as well! (b/c all distances 0)
        for i in gdat.indxsamp:
    
            #for specific catalog, find indices of start and end within large tree
            indxstarloww = np.sum(chan['numb'][:i])
            indxstarhigh = np.sum(chan['numb'][:i+1])
    
            #want in the form of a numpy array so we can use array slicing/masking  
            matches = np.array(matches)
    
            #find the locations of matches to star k within specific catalog i
            culled_matches =  matches[np.logical_and(matches >= indxstarloww, matches < indxstarhigh)] 
    
            if culled_matches.size > 0:
    
                #cut according to mask
                culled_matches = culled_matches[mask[culled_matches] == 0]
        
                #if there are matches remaining, we then find the brightest and update
                if culled_matches.size > 0:
    
                    #find brightest (first in the sorted catalogue)
                    match = np.min(culled_matches)
    
                    #flag it in the mask
                    mask[match] += 1
    
                    #add information to cluster array
                    featcond[i, k, gdat.indxxpos] = catlsort[i, match-indxstarloww, gdat.indxxpos]
                    featcond[i, k, gdat.indxypos] = catlsort[i, match-indxstarloww, gdat.indxypos]
                    featcond[i, k, gdat.indxflux] = catlsort[i, match-indxstarloww, gdat.indxflux]
                    prvl[k] += 1
    
    # generate condensed catalog from clusters
    numbsourseed = len(catlseed)
    
    #arrays to store condensed catalog parameters
    xposmean = np.zeros(numbsourseed)
    yposmean = np.zeros(numbsourseed)
    fluxmean = np.zeros((gdat.numbener, gdat.numbtime, numbsourseed))
    magtmean = np.zeros(numbsourseed)
    stdvxpos = np.zeros(numbsourseed)
    stdvypos = np.zeros(numbsourseed)
    stdvflux = np.zeros((gdat.numbener, gdat.numbtime, numbsourseed))
    stdvmagt = np.zeros(numbsourseed)
    
    # confidence interval defined for err_(x,y,f)
    pctlhigh = 84
    pctlloww = 16
    for i in indxsourseed:
        # only calculate means and variances for matches
        indxmtch = np.where(featcond[:, i, gdat.indxflux[0,0]] > 0)[0]
        xpos = featcond[indxmtch, i, gdat.indxxpos]
        ypos = featcond[indxmtch, i, gdat.indxypos]
        flux = featcond[:, i, gdat.indxflux][indxmtch, :]
        if (featcond[:, i, gdat.indxflux] - np.mean(featcond[:, i, gdat.indxflux]) == 0.).all():
            print 'Warning! condensed catalog flux uncertainties vanish!'

        assert xpos.size == ypos.size
        
        xposmean[i] = np.mean(xpos)
        yposmean[i] = np.mean(ypos)
        fluxmean[:, :, i] = np.mean(flux, axis=0)

        if xpos.size > 1:
            stdvxpos[i] = np.percentile(xpos, pctlhigh) - np.percentile(xpos, pctlloww)
            stdvypos[i] = np.percentile(ypos, pctlhigh) - np.percentile(ypos, pctlloww)
            stdvflux[:, :, i] = np.percentile(flux, pctlhigh, axis=0) - np.percentile(flux, pctlloww, axis=0)
            if (stdvflux[:, :, i] == 0.).any():
                print 'Warning! Samples of condensed catalog elements have vanishing variance! matches: '+str(xpos.size)

    catlcond = np.zeros((gdat.numbparacatlcond, numbsourseed, 2))
    catlcond[gdat.indxxpos, :, 0] = xposmean
    catlcond[gdat.indxxpos, :, 1] = stdvxpos
    catlcond[gdat.indxypos, :, 0] = yposmean
    catlcond[gdat.indxypos, :, 1] = stdvypos
    catlcond[gdat.indxflux, :, 0] = fluxmean
    catlcond[gdat.indxflux, :, 1] = stdvflux
    if gdat.strgmodl == 'galx':
        catlcond[gdat.indxxxmo, :, 0] = xxmomean
        catlcond[gdat.indxxxmo, :, 1] = stdvxxmo
    catlcond[gdat.indxprvl, :, 0] = prvl / float(gdat.numbsamp)
    catlcond[gdat.indxprvl, :, 1] = np.sqrt(prvl) / float(gdat.numbsamp)
    

    ## h5 file path
    print 'Will write the chain to %s...' % pathcatlcond
    filecatlcond = h5py.File(pathcatlcond, 'w')
    filecatlcond.create_dataset('catlcond', data=catlcond)
    filecatlcond.close()
    
    return catlcond


def retr_path():
    
    pathlion = os.environ['LION_PATH'] + '/'
    pathdata = os.environ['LION_DATA_PATH'] + '/'
    
    return pathlion, pathdata


# configurations

def read_datafromtext(strgdata):
    
    pathlion, pathdata = retr_path()

    filepixl = open(pathdata + strgdata + '_pixl.txt')
    filepixl.readline()
    bias, gain = [i for i in filepixl.readline().split()]
    filepixl.close()
    cntpdata = np.loadtxt(pathdata + strgdata + '_cntp.txt')
    cntpdata -= bias
    
    return cntpdata, bias, gain


def retr_catltrue(pathdata, strgmodl, strgdata):

    '''
    get the true catalog
    '''
    
    catltrue = {}
    if gdat.strgmodl == 'star':
        truth = np.loadtxt(pathdata + strgdata + '_true.txt')
        catltrue['numb'] = truth[:, 0].size
        catltrue['xpos'] = truth[:, 0]
        catltrue['ypos'] = truth[:, 1]
        catltrue['flux'] = truth[:, 2]
    
    if gdat.strgmodl == 'galx':
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
    
    return catltrue


def setp_varbvalu(gdat, strgvarb, valu, strgmodl=None, comm=False):
    
    if comm:
        liststrgmodl = ['']
    else:
        if strgmodl is None:
            if gdat.datatype == 'mock':
                liststrgmodl = ['true', 'fitt']
            else:
                liststrgmodl = ['fitt']
        else:
            liststrgmodl = [strgmodl]
    liststrgmodl = copy.deepcopy(liststrgmodl)
    for gdat.strgmodltemp in liststrgmodl:
        setp_varbiter(gdat, gdat.strgmodltemp, strgvarb, valu)


def setp_varbiter(gdat, strgmodltemp, strgvarbtemp, valu):
    
    try:
        valutemp = getattr(gdat, strgvarbtemp)
        if valutemp is None:
            raise
        setattr(gdat, gdat.strgmodltemp + strgvarbtemp, valutemp)
    except:
        try:
            valutemp = getattr(gdat, gdat.strgmodltemp + strgvarbtemp)
            if valutemp is None:
                raise
        except:
            setattr(gdat, gdat.strgmodltemp + strgvarbtemp, valu)
   

def cnfg_galx():
    
    strgdata = {}
    numbside = 10
    
    main( \
         #verbtype=2, \
         inittype='rand', \
         maxmnumbstar=200, \
         maxmnumbgalx=200, \
         diagmode=True, \
         datatype='mock', \
         boolplotinit=False, \
         strgmodl='galx'
        )


def cnfg_wise():
    
    # read the data
    #strgdata = 'sdss0921'
    
    #gdat = gdatstrt()
    
    #gdat.pathlion, pathdata = retr_path()
    
    # setup
    #setp(gdat)
    
    # read the data
    pathdata = os.environ['LION_DATA_PATH'] + '/unwise/unwise-1497p015-w1-img-m.fits'
    pathweig = os.environ['LION_DATA_PATH'] + '/unwise/unwise-1497p015-w1-std-m.fits'
    pathmask = os.environ['LION_DATA_PATH'] + '/unwise/unwise-1497p015-msk.fits.gz'
    pathpsfn = os.environ['LION_DATA_PATH'] + '/unwise/1497p015.1.info.fits'
    pathback = os.environ['LION_DATA_PATH'] + '/unwise/1497p015.1.mod.fits'
    print 'Reading %s...' % pathdata
    xmin, ymin, smalbnds = 896, 896, 128
    cntpdata = fits.open(pathdata)[0].data[None, ymin:ymin+smalbnds, xmin:xmin+smalbnds, None]
    cntpback = fits.open(pathback)[2].data[None, ymin:ymin+smalbnds, xmin:xmin+smalbnds, None]
    weig = 1. / fits.open(pathweig)[0].data[None, ymin:ymin+smalbnds, xmin:xmin+smalbnds, None]**2
    magtzero = 22.5 # if we were good people, we would read it in
    
    # read Cosmos catalog
    pathcosm = os.environ['LION_DATA_PATH'] + '/unwise/astrometry.scosmos_irac_04076395.fits'
    objtcosmcatl = fits.open(pathcosm)[1].data

    objtwcss = WCS(fits.open(pathdata)[0].header)
    arry = np.zeros((objtcosmcatl['ra'].size, 2))
    arry[:, 0] = objtcosmcatl['ra']
    arry[:, 1] = objtcosmcatl['dec']
    temp = objtwcss.all_world2pix(arry, 0)
    xposcosm = temp[:, 0] - xmin
    yposcosm = temp[:, 1] - ymin

    gdat = gdatstrt()
    gdat.numbener = 1
    gdat.indxener = [0]
    gdat.numbsidepsfnusam = 125
    cntppsfnorig = fits.open(pathpsfn)[3].data[None,:,:]
    cntppsfnusam = np.empty((gdat.numbener, gdat.numbsidepsfnusam, gdat.numbsidepsfnusam))
    for i in gdat.indxener:
        thispsfnorig = cntppsfnorig[i,:,:]
        origsize = cntppsfnorig.shape[1] # check if square?
        subssize = 25
        indxstrt = origsize/2 - subssize/2
        indxfnsh = origsize/2 + subssize/2 + 1
        
        subspsfn = thispsfnorig[indxstrt:indxfnsh,indxstrt:indxfnsh]
        cntppsfnusam[i,:,:] = scipy.misc.imresize(subspsfn, (125,125), interp='lanczos', mode='F')

    weig[~np.isfinite(weig)] = 0.
    maskwise = fits.open(pathmask)[0].data
    weig *= np.logical_or(maskwise == 0, maskwise == 64)[None,:smalbnds,:smalbnds,None] # is 0 too restrictive? 64 might be OK

    catlrefr = [{}, {}]
    # cosmos catalog
    maskcosm = (objtcosmcatl['flux_c1_4'] > 0) * (xposcosm > 0) * (xposcosm < smalbnds) * (yposcosm > 0) * (yposcosm < smalbnds)
    # take only objects with positive flux and within bounds
    catlrefr[0]['xpos'] = xposcosm[maskcosm]
    catlrefr[0]['ypos'] = yposcosm[maskcosm]
    catlrefr[0]['magt'] = (-2.5 * np.log10(1e-6 * objtcosmcatl['flux_c1_4'][maskcosm] / 3631) - 2.699)[None, None, :] # 3.6 um at 4.1, and AB->Vega conversion for W1
    catlrefr[0]['flux'] = 10**(-0.4 * (catlrefr[0]['magt']-magtzero))[None, None, :]
    # crowdsource catalog
    crsrdata = os.environ['LION_DATA_PATH'] + '/unwise/1497p015.1.cat.fits'
    crsrcatl = fits.open(crsrdata)[1].data
    #xy convention flipped in crowdsource catalog
    mask = (crsrcatl['x'] > ymin) * (crsrcatl['x'] < ymin+smalbnds) * (crsrcatl['y'] > xmin) * (crsrcatl['y'] < xmin+smalbnds)
    catlrefr[1]['xpos'] = crsrcatl['y'][mask] - xmin
    catlrefr[1]['ypos'] = crsrcatl['x'][mask] - ymin
    catlflux = (crsrcatl['flux'][mask])[None, None, :]
    catlrefr[1]['flux'] = catlflux
    catlrefr[1]['magt'] = magtzero - 2.5 * np.log10(catlflux)
   
    factusam = 5

    main( \
         cntpdata=cntpdata, \
         cntppsfnusam=cntppsfnusam, \
         factusam=factusam, \
         
         weig=weig, \

         catlrefr=catlrefr, \
         lablrefr=['Cosmos', 'CwdSrc'], \
         colrrefr=['orange', 'g'], \
        
         cntpback=cntpback, \
        
         fittminmflux=5., \

         #exprtype='wise', exprtype, \

         sizeregi=32, \
         marg=16,
         #numbswep=200, \
         #colrstyl='pcat', \
         numbplotfram=10, \
         boolplotinit=False, \
         #boolplotplot=False, \
         #diagmode=True, \
         magtzero=magtzero, \
         verbtype=1, \
         numbswep=1000, \
         #numbloop=1000, \
         priotype = 'prio', \
         #strgproc='pcat_20190301_143248_cnfg_wise_1000000'
        )


def cnfg_defa():
    
    # read the data
    #strgdata = 'sdss0921'
    #cntpdatatemp, bias, gain = read_datafromtext(strgdata)
    
    gdat = gdatstrt()
    
    gdat.pathlion, pathdata = retr_path()
    
    # setup
    #setp(gdat)
    
    # gain
    gain = 4.62
    
    # read the data
    strgdata = 'sdss0921_00010001_nomi'
    path = pathdata + strgdata + '_mock.h5'
    print 'Reading %s...' % path
    filetemp = h5py.File(path, 'r')
    cntpdata = filetemp['cntpdata'][()]
    #gain = filetemp['gain'][()]
    #truecatl = read_catl(gdat, path)
    #bias = 0.
    
    cntpback = np.zeros((cntpdata.shape[1], cntpdata.shape[2])) + 200.

    #liststrgener = ['rbnd']
    #read_psfn(pathdata, strgdata, liststrgener)
    #strgdata = 'sdss0921'
    #filepsfn = open(pathdata + strgdata + '_psfn.txt')
    #numbsidepsfn, factusam = [np.int32(i) for i in filepsfn.readline().split()]
    #filepsfn.close()
    #cntppsfn = np.loadtxt(pathdata + strgdata + '_psfn.txt', skiprows=1)
    #cntppsfn = cntppsfn[None, :, :] 
    #numbtime = 1
    #numbener = 1

    #numbsidexpos = 100
    #numbsideypos = 100

    #gdat.strgmodl = 'star'
    #datatype = 'mock'
    #
    #catlrefr = []

    #if datatype == 'mock':
    #    catlrefr.append(truecatl)
    #    #catlrefr[0]['flux'] = catlrefr[0]['flux'][None, None, :]
    #    lablrefr = ['True']
    #    colrrefr = ['g']
    #else:
    #    lablrefr = ['HST 606W']
    #    colrrefr = ['m']
   
    exprtype = 'sdss'
    
    bias = 0.
    gain = 1.

    main( \
         cntpdata=cntpdata, \
         #cntppsfn=cntppsfn, \
         #factusam=factusam, \

         #catlrefr=catlrefr, \
         #lablrefr=lablrefr, \
         #colrrefr=colrrefr, \
        
         #cntpback=cntpback, \
        
         exprtype=exprtype, \

         sizeregi=20, \
         #numbswep=200, \
         colrstyl='pcat', \
         #boolplotsave=False, \
         #diagmode=True, \
         bias=bias, \
         gain=gain, \
         #verbtype=2, \
         numbswep=100, \
         numbloop=1000, \
        )


def mainarry( \
             dictvarbvari, \
             dictvarb, \
             listnamecnfgextn, \
             forcneww=False, \
             forcprev=False, \
             strgpara=False, \
             execpara=True, \
             strgcnfgextnexec=None, \
             listnamevarbcomp=[], \
             listscalvarbcomp=[], \
             listlablvarbcomp=[], \
             listtypevarbcomp=[], \
             listpdfnvarbcomp=[], \
             listgdatvarbcomp=[], \
             namexaxi=None, \
             lablxaxi=None, \
             listtickxaxi=None, \
             scalxaxi=None, \
            ):
    
    print 'Running Lion in array mode...'
    
    numbswep = len(dictvarbvari)
    indxswep = np.arange(numbswep)

    cntrcomp = 0
    
    if execpara:
        cntrproc = 0

    listrtag = []
    listpridchld = []
    for k, strgcnfgextn in enumerate(listnamecnfgextn):
        
        if strgcnfgextnexec != None:
            if strgcnfgextn != strgcnfgextnexec:
                continue
        
        strgcnfg = inspect.stack()[1][3] + '_' + strgcnfgextn
    
        print 'Configuration: '
        print strgcnfg 

        dictvarbtemp = copy.deepcopy(dictvarb)
        for strgvarb, valu in dictvarbvari[strgcnfgextn].iteritems():
            dictvarbtemp[strgvarb] = valu
        dictvarbtemp['strgcnfg'] = strgcnfg
    
        cntrcomp += 1
        # temp
        if execpara and strgcnfgextnexec is None:
            cntrproc += 1
            prid = os.fork()
            if prid > 0:
                listpridchld.append(prid)
            else:
                print 'Forking a child process to run the configuration extension...' 
                rtag = main(**dictvarbtemp)
                #if 'checprio' in dictvarbtemp and dictvarbtemp['checprio']:
                os._exit(0)
        else:
            print 'Calling the main Lion function without forking a child...' 
            listrtag.append(main(**dictvarbtemp))
    
    if execpara and strgcnfgextnexec is None:
        for prid in listpridchld:
            os.waitpid(prid, 0)
        if cntrproc > 0:
            print 'Exiting before comparion plots because of parallel execution...'
            return
    
    if cntrcomp == 0:
        print 'Found no runs...'

    strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if strgcnfgextnexec != None or namexaxi is None: 
        return
    
    print
    print 'Making comparison plots...'
     
    if 'mockonly' in dictvarb and dictvarb['mockonly']:
        listgdat = retr_listgdat(listrtag, typegdat='init')
    else:
        listgdat = retr_listgdat(listrtag)
    
    numbgdat = len(listgdat)

    for namevarbscal in listgdat[0].listnamevarbscal:
        booltemp = True
        for k in range(1, numbgdat - 1):
            if not namevarbscal in listgdat[k].listnamevarbscal:
                booltemp = False
        if booltemp:
            if namevarbscal in listnamevarbcomp:
                raise Exception('')
            listnamevarbcomp += [namevarbscal]
            listscalvarbcomp += [getattr(listgdat[0], 'scal' + namevarbscal)]
            listlablvarbcomp += [getattr(listgdat[0], 'labl' + namevarbscal + 'totl')]
            listtypevarbcomp += ['pctl']
            listpdfnvarbcomp += ['post']
            listgdatvarbcomp += ['post']
    
    # add others to the variable list
    listnamevarbcomp += ['lliktotl', 'lliktotl', 'infopost', 'bcom', 'lliktotl', 'lliktotl', 'lliktotl', 'levipost']
    listscalvarbcomp += ['self', 'self', 'self', 'self', 'self', 'self', 'self', 'self']
    listlablvarbcomp += ['$\ln P(D|M_{min})$', '$\ln P(D|M_{max})$', '$D_{KL}$', '$\eta_B$', '$\sigma_{P(D|M)}$', r'$\gamma_{P(D|M)}$', r'$\kappa_{P(D|M)}$', \
                                                                                                                                                       '$\ln P_H(D)$']
    listtypevarbcomp += ['minm', 'maxm', '', '', 'stdv', 'skew', 'kurt', '']
    listpdfnvarbcomp += ['post', 'post', 'post', 'post', 'post', 'post', 'post', 'post']
    listgdatvarbcomp += ['post', 'post', 'post', 'post', 'post', 'post', 'post', 'post']
    
    arrytemp = array([len(listnamevarbcomp), len(listscalvarbcomp), len(listlablvarbcomp), len(listtypevarbcomp), len(listpdfnvarbcomp), len(listgdatvarbcomp)])
    if (arrytemp - mean(arrytemp) != 0.).all():
        raise Exception('')

    # add log-evidence to the variable list, if prior is also sampled
    booltemp = True
    for k in range(numbgdat):
        if not listgdat[k].checprio:
            booltemp = False
    
    pathbase = '%s/imag/%s_%s/' % (os.environ["LION_DATA_PATH"], strgtimestmp, inspect.stack()[1][3])
    if booltemp:
        listgdatprio = retr_listgdat(listrtag, typegdat='finlprio')
        
        listnamevarbcomp += ['leviprio']
        listscalvarbcomp += ['self']
        listlablvarbcomp += ['$\ln P_{pr}(D)$']
        listtypevarbcomp += ['']
        listpdfnvarbcomp += ['prio']
        listgdatvarbcomp += ['prio']
    
    # time stamp
    strgtimestmp = tdpy.util.retr_strgtimestmp()
    
    dictoutp = dict()
    liststrgvarbtotl = []
    for (typevarbcomp, pdfnvarbcomp, namevarbcomp) in zip(listtypevarbcomp, listpdfnvarbcomp, listnamevarbcomp):
        strgtemp = typevarbcomp + pdfnvarbcomp + namevarbcomp
        liststrgvarbtotl.append(strgtemp)
        dictoutp[strgtemp] = [[] for k in range(numbiter)]
    
    for k in indxswep:
        for a, strgvarbtotl in enumerate(liststrgvarbtotl):
            if listgdatvarbcomp[a] == 'prio':
                gdattemp = listgdatprio[k]
            else:
                gdattemp = listgdat[k]
            dictoutp[strgvarbtotl][k] = getattr(gdattemp, strgvarbtotl)

    cmnd = 'mkdir -p %s' % pathbase 
    os.system(cmnd)
    
    print 'dictoutp'
    print dictoutp
    print 'listnamevarbcomp'
    print listnamevarbcomp
    cmnd = 'gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=%smrgd.pdf' % pathbase
    for strgvarbtotl, varboutp in dictoutp.iteritems():
        
        figr, axis = plt.subplots(figsize=(6, 6))
        ydat = empty(numbiter)
        yerr = zeros((2, numbiter))
            
        indxlist = liststrgvarbtotl.index(strgvarbtotl)
        
        if listscalvarbcomp is None:
            scalyaxi = getattr(listgdat[0], 'scal' + listnamevarbcomp[indxlist])
        else:
            scalyaxi = listscalvarbcomp[indxlist]
        
        lablyaxi = listlablvarbcomp[indxlist]
        
        try:
            factplot = getattr(listgdat[0], 'fact' + listnamevarbcomp[indxlist] + 'plot')
        except:
            factplot = 1.
        
        try:
            if listtypevarbcomp[indxlist] == 'pctl':
                trueyaxi = getattr(listgdat[0], 'true' + listnamevarbcomp[indxlist])
            else:
                trueyaxi = getattr(listgdat[0], 'true' + listtypevarbcomp[indxlist] + listnamevarbcomp[indxlist])
        except:
            trueyaxi = None
        
        for k in indxswep:
            
            if isinstance(varboutp[k], list) or isinstance(varboutp[k], ndarray) and varboutp[k].ndim > 2:
                raise Exception('')
            elif isinstance(varboutp[k], float):
                ydat[k] = varboutp[k]
            else:
                if listtypevarbcomp[indxlist] != 'pctl':
                    yerr[:, k] = 0.
                if varboutp[k].ndim == 2:
                    if varboutp[k].shape[1] != 1:
                        raise Exception('varboutp format is wrong.')
                    varboutp[k] = varboutp[k][:, 0]
                    if listtypevarbcomp[indxlist] == 'pctl':
                        yerr[:, k] = getattr(listgdat[k], 'errr' + listpdfnvarbcomp[indxlist] + listnamevarbcomp[indxlist])[:, 0]
                else:
                    if listtypevarbcomp[indxlist] == 'pctl':
                        yerr[:, k] = getattr(listgdat[k], 'errr' + listpdfnvarbcomp[indxlist] + listnamevarbcomp[indxlist])
                ydat[k] = varboutp[k][0]
        
        print 'listnamevarbcomp[indxlist]'
        print listnamevarbcomp[indxlist]
        print 'lablyaxi'
        print lablyaxi
        print 'ydat'
        print ydat
        print 'yerr'
        print yerr

        axis.errorbar(indxswep+1., ydat * factplot, yerr=yerr * factplot, color='b', ls='', markersize=15, marker='o', lw=3)
        indxrtagyerr = where((yerr[0, :] > 0.) | (yerr[1, :] > 0.))[0]
        if indxrtagyerr.size > 0:
            temp, listcaps, temp = axis.errorbar(indxswep[indxrtagyerr]+1., ydat[indxrtagyerr] * factplot, yerr=yerr[:, indxrtagyerr] * factplot, \
                                                                                color='b', ls='', capsize=15, markersize=15, marker='o', lw=3)
            for caps in listcaps:
                caps.set_markeredgewidth(3)
        
        if trueyaxi != None:
            axis.axhline(trueyaxi, ls='--', color='g')
        
        if lablxaxi is None:
            lablxaxi = getattr(listgdat[0], 'labl' + namexaxi + 'totl')
        
        if scalxaxi is None:
            scalxaxi = getattr(listgdat[0], 'scal' + namexaxi)
        
        axis.set_xlabel(lablxaxi)
        axis.set_xticks(indxswep+1.)
        axis.set_xticklabels(listtickxaxi)
        
        axis.set_ylabel(lablyaxi)
        if scalyaxi == 'logt':
            axis.set_yscale('log')
        plt.tight_layout()
        
        pathfull = '%s%s_%s_%s.pdf' % (pathbase, strgtimestmp, inspect.stack()[1][3], liststrgvarbtotl[indxlist])
        print 'Writing to %s...' % pathfull
        print
        print
        print
        plt.tight_layout()
        plt.savefig(pathfull)
        plt.close(figr)
    
        cmnd += ' %s' % pathfull

    print cmnd
    os.system(cmnd)

    #print 'Making animations...'
    #for rtag in listrtag:
    #    print 'Working on %s...' % rtag
    #    proc_anim(rtag=rtag)
    
    print 'Compiling run plots...'
    cmnd = 'python comp_rtag.py'
    for rtag in listrtag: 
        cmnd += ' %s' % rtag
    os.system(cmnd)

    return listrtag


def retr_cntppsfnusam(gdat):

    filepsfn = open(gdat.pathdata + 'idR-002583-2-0136-psfg.txt')
    gdat.numbsidepsfn, gdat.factusam = [np.int32(i) for i in filepsfn.readline().split()]
    filepsfn.close()
    
    gdat.numbsidepsfnusam = gdat.numbsidepsfn * gdat.factusam
    gdat.numbpixlpsfn = gdat.numbsidepsfn**2
    gdat.cntppsfnusam = np.empty((gdat.numbener, gdat.numbsidepsfnusam, gdat.numbsidepsfnusam))
    gdat.cntppsfnusam[0, :, :] = np.loadtxt(gdat.pathdata + 'idR-002583-2-0136-psfg.txt', skiprows=1)
    if gdat.numbener > 1:
        gdat.cntppsfnusam[1, :, :] = np.loadtxt(pathdata + 'idR-002583-2-0136-psfr.txt', skiprows=1)
        gdat.cntppsfnusam[2, :, :] = np.loadtxt(pathdata + 'idR-002583-2-0136-psfi.txt', skiprows=1)
        

def cnfg_time(strgcnfgextnexec=None):
    
    dictargs = {}
    
    #dictargs['factusam'] = factusam
    dictargs['numbener'] = 1
    dictargs['numbtime'] = 3
    
    #dictargs['stdvposiprop'] = 1e-100
    #dictargs['stdvfluxprop'] = 1e-100
    
    #dictargs['cntpback'] = 225.
    dictargs['truebias'] = 0.
    #dictargs['numbswepburn'] = 40
    
    dictargs['truenumbstar'] = 40
    #dictargs['truenumbstar'] = 3
    
    dictargs['inittype'] = 'rand'
    dictargs['numbloop'] = 100
    dictargs['numbswep'] = 100
    dictargs['numbplotfram'] = 2
    dictargs['trueminmflux'] = 100.
    dictargs['fittminmflux'] = 100.
    
    #dictargs['maxmnumbstar'] = 1
    
    dictargs['numbsidexpos'] = 10
    dictargs['numbsideypos'] = 10
    
    dictargs['boolspre'] = False
    dictargs['boolclib'] = False
    #dictargs['verbtype'] = 2
    
    dictargs['boolbili'] = True
    
    dictargs['diagmode'] = True
    dictargs['colrstyl'] = 'pcat'
    #dictargs['strgproc'] = 'pcat_20180714_142316_cnfg_time_000200'
    dictargs['strgmode'] = 'pcat'
    dictargs['probprop'] = np.array([2., 1., 1.])
    dictargs['booltile'] = False
    #dictargs['sizeregi'] = gdat.sizeregi
    #dictargs['catlrefr'] = catlrefr
    dictargs['colrrefr'] = ['g']
    dictargs['lablrefr'] = ['Mock']
    #dictargs['strgstat'] = 'test'
         
    #dictargs['numbener'] = 1

    listnamecnfgextn = ['defa', 'sing', 'nomi', 'crow', 'timehigh', 'crowplot', 'timehighcent', 'timehighcentcrow']
    #listnamecnfgextn = ['nomi', 'crow', 'timehigh']
    dictargsvari = {}
    for namecnfgextn in listnamecnfgextn:
        dictargsvari[namecnfgextn] = {}
    
    dictargsvari['defa']['truenumbstar'] = 1
    dictargsvari['defa']['initxpos'] = np.array([4.5])
    dictargsvari['defa']['initypos'] = np.array([4.5])
    dictargsvari['defa']['maxmnumbstar'] = 20
    dictargsvari['defa']['trueminmflux'] = 1e3
    dictargsvari['defa']['numbtime'] = 1

    #dictargsvari['nomi']['strgproc'] = 'pcat_20180718_125001_cnfg_time_nomi_010000'
    dictargsvari['nomi']['truenumbstar'] = 3
    dictargsvari['nomi']['maxmnumbstar'] = 20
    dictargsvari['nomi']['trueminmflux'] = 1e3
    dictargsvari['nomi']['trueminmflux'] = 1e3
    dictargsvari['nomi']['indxdatatotl'] = np.array([3, 5, 6])
    labldata = np.zeros((1, 3), dtype=object)
    for t in range(3):
        labldata[0, t] = 'Time bin %d' % t
    labldataaxis = np.zeros((1, 3), dtype=object)
    for t in range(3):
        labldata[0, t] = 'Time bin %d' % t
        labldata[0, t] = 'Time bin %d' % t
    dictargsvari['nomi']['indxdatatotl'] = np.arange(3, 6)
    
    dictargsvari['sing']['truenumbstar'] = 1
    dictargsvari['sing']['initxpos'] = np.array([4.5])
    dictargsvari['sing']['initypos'] = np.array([4.5])
    dictargsvari['sing']['maxmnumbstar'] = 20
    dictargsvari['sing']['trueminmflux'] = 1e4
    #dictargsvari['sing']['strgproc'] = 'pcat_20180718_125000_cnfg_time_sing_010000'
    dictargsvari['sing']['labldata'] = labldata
    
    dictargsvari['crow']['numbstar'] = 100
    dictargsvari['crow']['numbsidexpos'] = 40
    dictargsvari['crow']['numbsideypos'] = 40
    #dictargsvari['crow']['strgproc'] = 'pcat_20180718_125003_cnfg_time_crow_010000'

    dictargsvari['crowplot']['numbplotfram'] = 50
    dictargsvari['crowplot']['numbswep'] = 200
    dictargsvari['crowplot']['numbstar'] = 100
    dictargsvari['crowplot']['numbsidexpos'] = 40
    dictargsvari['crowplot']['numbsideypos'] = 40

    dictargsvari['timehigh']['truenumbstar'] = 4
    dictargsvari['timehigh']['trueminmflux'] = 1e6
    dictargsvari['timehigh']['numbtime'] = 30
    dictargsvari['timehigh']['maxmnumbstar'] = 4
    
    dictargsvari['timehighcent']['numbswep'] = 1000
    dictargsvari['timehighcent']['truefluxcent'] = np.random.randn(20).reshape((1, 20)) * 1e-2 + 1.72e5
    dictargsvari['timehighcent']['cntpback'] = np.full([11, 11], 1e2 * 3600.)
    dictargsvari['timehighcent']['numbloop'] = 1
    dictargsvari['timehighcent']['truenumbstar'] = 1
    dictargsvari['timehighcent']['trueminmflux'] = 1e4
    dictargsvari['timehighcent']['fittminmflux'] = 1e0
    dictargsvari['timehighcent']['numbtime'] = 20
    dictargsvari['timehighcent']['numbsidexpos'] = 11
    dictargsvari['timehighcent']['numbsideypos'] = 11
    dictargsvari['timehighcent']['maxmnumbstar'] = 4
    dictargsvari['timehighcent']['inittype'] = None
    
    dictargsvari['timehighcentcrow']['numbswep'] = 1000
    dictargsvari['timehighcentcrow']['truefluxcent'] = np.random.randn(20).reshape((1, 20)) * 1e2 + 1e5
    dictargsvari['timehighcentcrow']['cntpback'] = np.full([11, 11], 1e2 * 3600.)
    dictargsvari['timehighcentcrow']['numbloop'] = 1
    dictargsvari['timehighcentcrow']['truenumbstar'] = 10
    dictargsvari['timehighcentcrow']['trueminmflux'] = 3e4
    dictargsvari['timehighcentcrow']['fittminmflux'] = 1e0
    dictargsvari['timehighcentcrow']['numbtime'] = 20
    dictargsvari['timehighcentcrow']['numbsidexpos'] = 11
    dictargsvari['timehighcentcrow']['numbsideypos'] = 11
    dictargsvari['timehighcentcrow']['maxmnumbstar'] = 20
    dictargsvari['timehighcentcrow']['inittype'] = None
    
    dictglob = mainarry( \
                        dictargsvari, \
                        dictargs, \
                        listnamecnfgextn, \
                        strgcnfgextnexec=strgcnfgextnexec, \
                       )


def cnfg_ener():
    
    gdat = gdatstrt()
    
    gdat.pathlion, pathdata = retr_path()
    
    setp(gdat)
    
    # read the data
    strgdata = 'sdss0921_00030001'
    #strgdata = 'sdss0921_00020001'

    path = pathdata + strgdata + '_mock.h5'
    print 'Reading %s...' % path
    filetemp = h5py.File(path, 'r')
    cntpdata = filetemp['cntpdata'][()]
    truegain = filetemp['gain'][()]
    truecatl = read_catl(gdat, path)
    
    gdat.numbener = cntpdata.shape[0]
    numbsidexpos = cntpdata.shape[2]
    numbsideypos = cntpdata.shape[1]
    numbpixl = numbsidexpos * numbsideypos
    print 'cntpdata'
    summgene(cntpdata)
    print 'numbpixl'
    print numbpixl
    
    strgmode = 'pcat'
    #strgmode = 'forc'
    gdat.strgmodl = 'star'
    
    #catlinit = retr_catltrue(pathdata, gdat.strgmodl, strgdata)
    # temp
    #indx = np.argsort(truecatl['flux'][0, 0, :])
    #truecatl['numb'] = 1000
    #truecatl['xpos'] = truecatl['xpos'][indx][:1000]
    #truecatl['ypos'] = truecatl['ypos'][indx][:1000]
    #truecatl['flux'] = truecatl['flux'][:, :, indx][:, :, :1000]
    truebias = 0.

    # read PSF
    #filepsfn = open(pathdata + 'idR-002583-2-0136-psfg.txt')
    #numbsidepsfn, factusam = [np.int32(i) for i in filepsfn.readline().split()]
    #filepsfn.close()
    #numbsidepsfnusam = numbsidepsfn * factusam
    #cntppsfn = np.empty((gdat.numbener, numbsidepsfnusam, numbsidepsfnusam))
    #cntppsfn[0, :, :] = np.loadtxt(pathdata + 'idR-002583-2-0136-psfg.txt', skiprows=1)
    
    gdat.indxener = np.arange(gdat.numbener)
    for i in gdat.indxener:
        if np.amax(cntppsfn[i, :, :]) == 0:
            raise Exception('')

    main( \
         cntpdata=cntpdata, \
         cntppsfn=cntppsfn, \
         factusam=factusam, \
         #numbswep=1, \
         #numbloop=2, \
         #maxmnumbstar=1, \
         numbswep=10, \
         numbloop=1000, \
         diagmode=True, \
         #verbtype=2, \
         colrstyl='pcat', \
        
         strgdata=strgdata, \

         probprop=np.array([0., 0.0, 1.]), \
         
         #boolplotsave=False, \
        
         strgmode=strgmode, \

         catlrefr=catlrefr, \
         #catlinit=catlinit, \
         #strgstat='test', \
         colrrefr=['g'], \
         lablrefr=['Mock'], \

         back=back, \
         
         bias=truebias, \
        )
         
#import astroquery
#from astroquery.simbad import Simbad
#result_table = Simbad.query_object("m1")
#print(result_table)
         
if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    else:
        globals().get(sys.argv[1])(*sys.argv[2:])


#import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double
