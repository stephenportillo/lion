from __init__ import *    
from main import *
import numpy as np
import os

gdat = gdatstrt()

np.random.seed(0)

gdat.diagmode = False

print 'Generating mock data...'

gdat.stdvlcpr = 1e-6
#gdat.stdvcolr = np.array([0.05])
#gdat.meancolr = np.array([0.])
gdat.stdvcolr = np.array([0.05, 0.05])
gdat.meancolr = np.array([0., 0.])
slop = np.float32(2.0)
minf = np.float32(100.)
gdat.numbstar = 200
sizeimag = [40, 40]

listdims = [[1, 1], [3, 1], [1, 10]]
strgdata = 'sdss0921'
strgpsfn = 'sdss0921'
gdat.pathlion = os.environ['LION_PATH'] + '/'
pathdata = os.environ['LION_DATA_PATH'] + '/'
    
# setup
gdat.boolplotsave = False 
setp(gdat)
setp_clib(gdat, gdat.pathlion)
gdat.verbtype = 1
    
for numbener, numbtime in listdims:
    
    gdat.numbener = numbener
    gdat.numbtime = numbtime
    gdat.indxener = np.arange(gdat.numbener)
    gdat.indxtime = np.arange(gdat.numbtime)
    gdat.numbcolr = gdat.numbener - 1
    gdat.numblcpr = gdat.numbtime - 1
    
    print 'numbener'
    print numbener
    print 'numbtime'
    print numbtime

    f = open(pathdata + strgpsfn + '_psfn.txt')
    nc, nbin = [np.int32(i) for i in f.readline().split()]
    f.close()
    # temp
    psf = np.loadtxt(pathdata + strgpsfn + '_psfn.txt', skiprows=1).astype(np.float32)
    psf = psf[None, :, :]    
    coefspix = psf_poly_fit(gdat, psf, nbin)
    
    print 'coefspix'
    summgene(coefspix)

    indxstar = np.arange(gdat.numbstar)
    
    # background
    trueback = np.float32(179.)
    
    # gain
    gain = np.float32(4.62)
    
    # position
    xpos = (np.random.uniform(size=gdat.numbstar)*(sizeimag[0]-1)).astype(np.float32)
    ypos = (np.random.uniform(size=gdat.numbstar)*(sizeimag[1]-1)).astype(np.float32)
    
    # flux
    fluxsumm = minf * np.exp(np.random.exponential(scale=1./(slop-1.), size=gdat.numbstar).astype(np.float32))
    
    flux = np.empty((gdat.numbener, gdat.numbtime, gdat.numbstar), dtype=np.float32)
    
    # temp
    flux[0, 0, :] = fluxsumm
    
    if gdat.numbener > 1:
        # temporal parameters
        colr = gdat.stdvcolr[:, None] * np.random.randn(gdat.numbcolr * gdat.numbstar).reshape((gdat.numbcolr, gdat.numbstar)).astype(np.float32) + gdat.meancolr[:, None]
        flux[1:, :, :] = fluxsumm[None, None, :] * 10**(0.4*colr[:, None, :])
    
    if gdat.numbtime > 1:
        # temporal parameters
        temp = np.random.random((gdat.numblcpr * gdat.numbstar)).reshape((gdat.numblcpr, gdat.numbstar)).astype(np.float32)
        temp = np.sort(temp, axis=0)
        temptemp = np.concatenate([np.zeros((1, gdat.numbstar), dtype=np.float32)] + [temp] + [np.ones((1, gdat.numbstar), dtype=np.float32)], axis=0)
        difftemp = temptemp[1:, :] - temptemp[:-1, :]
        
        #for k in range(10):
        #    if (np.sum(difftemp[:, k]) != 1).any():
        #        print 'temp[:, k]'
        #        print temp[:, k]
        #        print 'temptemp[:, k]'
        #        print temptemp[:, k]
        #        print 'np.sum(difftemp[:, k])'
        #        print np.sum(difftemp[:, k])
        #
        #assert (np.sum(difftemp, axis=0) == 1.).all()

        flux[:, :, :] = fluxsumm[None, None, :] * difftemp[None, :, :]
        
        # inject transits
        indxstartran = np.random.choice(indxstar, size=gdat.numbstar/2, replace=False)
        for k in indxstartran:
            indxinit = np.random.choice(gdat.indxtime)
            indxtemp = np.arange(indxinit, indxinit + 4) % gdat.numblcpr
            flux[:, indxtemp, k] = np.random.rand()
    
        #flux[:, 1:, :] = fluxsumm[None, None, :] * lcpr[None, :, :]
    
    print 'flux'
    summgene(flux)
    # evaluate model
    cntpdata = eval_modl(gdat, xpos, ypos, flux, trueback, nc, coefspix, clib=gdat.clib.clib_eval_modl, sizeimag=sizeimag)
    
    if not np.isfinite(cntpdata).all():
        raise Exception('')
    
    # temp -- why negative?
    cntpdata[cntpdata < 1] = 1.
    
    # add noise
    vari = cntpdata / gain
    print 'vari'
    summgene(vari)
    cntpdata += (np.sqrt(vari) * np.random.normal(size=(gdat.numbener, sizeimag[1], sizeimag[0], gdat.numbtime))).astype(np.float32)
    
    print 'cntpdata'
    summgene(cntpdata)
    if not np.isfinite(cntpdata).all():
        raise Exception('')
    
    # write to file
    path = pathdata + strgdata + '_%04d%04d_mock.h5' % (gdat.numbener, gdat.numbtime)
    print 'Writing to %s...' % path
    filetemp = h5py.File(path, 'w')
    filetemp.create_dataset('cntpdata', data=cntpdata)
    filetemp.create_dataset('numb', data=gdat.numbstar)
    filetemp.create_dataset('xpos', data=xpos)
    filetemp.create_dataset('ypos', data=ypos)
    filetemp.create_dataset('flux', data=flux)
    filetemp.create_dataset('gain', data=gain)
    filetemp.close()
    
    print


