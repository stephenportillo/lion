import numpy as np
import matplotlib.pyplot as plt
from __init__ import *

def psf_poly_fit(psfnusam, factusam):
    
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
            print 'A'
            summgene(A)
            print 'p'
            summgene(p)
            print 'coefspix[:, i, j]'
            summgene(coefspix[:, i, j])
            print
            coefspix[:, i, j] = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, p)) 
    coefspix = coefspix.reshape(coefspix.shape[0], coefspix.shape[1] * coefspix.shape[2])
    
    return coefspix


def image_model_eval(x, y, f, back, imsz, numbpixlpsfnside, coefspix, numbtime, booltimebins, lcpreval, \
                     regsize=None, margin=0, offsetx=0, offsety=0, weig=None, ref=None, lib=None):
    
    assert x.dtype == np.float32
    assert y.dtype == np.float32
    assert f.dtype == np.float32
    assert coefspix.dtype == np.float32
    if ref is not None:
        assert ref.dtype == np.float32
    
    numbpixlpsfn = numbpixlpsfnside**2

    numbparaspix = coefspix.shape[0]

    if weig is None:
        weig = np.full(imsz, 1., dtype=np.float32)
    if regsize is None:
        regsize = max(imsz[0], imsz[1])

    # FIXME sometimes phonions are outside image... what is best way to handle?
    goodsrc = (x > 0) * (x < imsz[0] - 1) * (y > 0) * (y < imsz[1] - 1)
    x = x.compress(goodsrc)
    y = y.compress(goodsrc)
    f = f.compress(goodsrc)

    numbphon = x.size
    rad = numbpixlpsfnside / 2 # 12 for numbpixlpsfnside = 25

    numbregiyaxi = imsz[1] / regsize + 1 # assumes imsz % regsize = 0?
    numbregixaxi = imsz[0] / regsize + 1

    ix = np.ceil(x).astype(np.int32)
    dx = ix - x
    iy = np.ceil(y).astype(np.int32)
    dy = iy - y

    dd = np.column_stack((np.full(numbphon, 1., dtype=np.float32), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, dx*dx*dy, dx*dy*dy, dy*dy*dy)).astype(np.float32) * f[:, None]

    if lib is None:
        
        if True:
            print 'lib is None'

        modl = np.full((imsz[1]+2*rad+1,imsz[0]+2*rad+1), back, dtype=np.float32)
        recon2 = np.dot(dd, coefspix).reshape((numbphon,numbpixlpsfnside,numbpixlpsfnside))
        recon = np.zeros((numbphon,numbpixlpsfnside,numbpixlpsfnside), dtype=np.float32)
        recon[:,:,:] = recon2[:,:,:]
        for i in xrange(numbphon):
            modl[iy[i]:iy[i]+rad+rad+1,ix[i]:ix[i]+rad+rad+1] += recon[i,:,:]

        modl = modl[rad:imsz[1]+rad,rad:imsz[0]+rad]

        if ref is not None:
                diff = ref - modl
        diff2 = np.zeros((numbregiyaxi, numbregixaxi), dtype=np.float64)
        for i in xrange(numbregiyaxi):
            y0 = max(i*regsize - offsety - margin, 0)
            y1 = min((i+1)*regsize - offsety + margin, imsz[1])
            for j in xrange(numbregixaxi):
                x0 = max(j*regsize - offsetx - margin, 0)
                x1 = min((j+1)*regsize - offsetx + margin, imsz[0])
                subdiff = diff[y0:y1,x0:x1]
                diff2[i,j] = np.sum(subdiff*subdiff*weig[y0:y1,x0:x1])
    else:
        recon = np.zeros((numbphon, numbpixlpsfn), dtype=np.float32)
        reftemp = ref
        if ref is None:
            if booltimebins:
                reftemp = np.zeros((numbtime, imsz[1], imsz[0]), dtype=np.float32)
            else:
                reftemp = np.zeros((imsz[1], imsz[0]), dtype=np.float32)
        if booltimebins:
            modl = np.full((numbtime, imsz[1], imsz[0]), back, dtype=np.float32)
            diff2 = np.zeros((numbtime, numbregiyaxi, numbregixaxi), dtype=np.float64)
        else:
            modl = np.full((imsz[1], imsz[0]), back, dtype=np.float32)
            diff2 = np.zeros((numbregiyaxi, numbregixaxi), dtype=np.float64)
        
        if True:
        #if False:
            print 'dd'
            summgene(dd)
            print 'coefspix'
            summgene(coefspix)
            print 'recon'
            summgene(recon)
            print 'modl'
            summgene(modl)
            print 'weig'
            print weig.shape
            print 'reftemp'
            print reftemp.shape
            print 'diff2'
            print diff2.shape
            print 'lcpreval'
            summgene(lcpreval)
            print 'booltimebins'
            print booltimebins
            print type(booltimebins)
            print 'numbtime'
            print numbtime
            print type(numbtime)
            print

        lib(imsz[0], imsz[1], numbphon, numbpixlpsfnside, numbparaspix, dd, coefspix, recon, ix, iy, modl, \
                                                reftemp, weig, diff2, regsize, margin, offsetx, offsety, numbtime, booltimebins, lcpreval)

    if ref is not None:
        return modl, diff2
    else:
        return modl
