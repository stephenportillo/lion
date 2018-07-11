from __init__ import *    
from main import *
import numpy as np
import os

gdat = gdatstrt()
gdatnotp = gdatstrt()

gdat.boolspre = True
np.random.seed(0)

gdat.diagmode = False
gdat.booltile = True

print 'Generating mock data...'

for strgtype in ['sing', 'nomi']:
    
    if strgtype == 'sing':
        boolcent = True
        minf = np.float32(10000.)
        gdat.numbstar = 1
    else:
        boolcent = False
        minf = np.float32(200.)
        gdat.numbstar = 200
    gdat.stdvlcpr = 1e-6
    #gdat.stdvcolr = np.array([0.05])
    #gdat.meancolr = np.array([0.])
    gdat.stdvcolr = np.array([0.05, 0.05])
    gdat.meancolr = np.array([0., 0.])
    slop = np.float32(2.0)
    sizeimag = [100, 100]
    
    if boolcent and gdat.numbstar != 1:
        raise Exception('')
    
    listdims = [[1, 1], [3, 1], [1, 3], [2, 2]]
    strgdata = 'sdss0921'
    strgpsfn = 'sdss0921'
    gdat.pathlion = os.environ['LION_PATH'] + '/'
    pathdata = os.environ['LION_DATA_PATH'] + '/'
        
    # setup
    gdat.boolplotsave = False 
    setp(gdat)
    setp_clib(gdat, gdatnotp, gdat.pathlion)
    gdat.verbtype = 1
        
    # get the 3-band PSF
    filepsfn = open(pathdata + 'idR-002583-2-0136-psfg.txt')
    numbsidepsfn, factusam = [np.int32(i) for i in filepsfn.readline().split()]
    filepsfn.close()
    numbsidepsfnusam = numbsidepsfn * factusam
    gdat.cntppsfn = np.empty((3, numbsidepsfnusam, numbsidepsfnusam))
    gdat.cntppsfn[0, :, :] = np.loadtxt(pathdata + 'idR-002583-2-0136-psfg.txt', skiprows=1).astype(np.float32)
    gdat.cntppsfn[1, :, :] = np.loadtxt(pathdata + 'idR-002583-2-0136-psfr.txt', skiprows=1).astype(np.float32)
    gdat.cntppsfn[2, :, :] = np.loadtxt(pathdata + 'idR-002583-2-0136-psfi.txt', skiprows=1).astype(np.float32)
    
    for numbener, numbtime in listdims:
        
        if numbener > 3:
            raise Exception('')
    
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
    
        cntppsfn = gdat.cntppsfn[:gdat.numbener, :, :]
    
        coefspix = psf_poly_fit(gdat, cntppsfn, factusam)
        
        print 'coefspix'
        summgene(coefspix)
    
        indxstar = np.arange(gdat.numbstar)
        
        # background
        trueback = np.float32(179.)
        
        # gain
        gain = np.float32(4.62)
        
        # position
        if boolcent:
            xpos = (np.array([0.5]) * (sizeimag[0] - 1)).astype(np.float32)
            ypos = (np.array([0.5]) * (sizeimag[0] - 1)).astype(np.float32)
        else:
            xpos = (np.random.uniform(size=gdat.numbstar)*(sizeimag[0]-1)).astype(np.float32)
            ypos = (np.random.uniform(size=gdat.numbstar)*(sizeimag[1]-1)).astype(np.float32)
        
        # flux
        fluxsumm = minf * np.exp(np.random.exponential(scale=1./(slop-1.), size=gdat.numbstar).astype(np.float32))
        
        flux = np.ones((gdat.numbener, gdat.numbtime, gdat.numbstar), dtype=np.float32)
        
        # temp
        flux *= fluxsumm
        
        if gdat.numbener > 1:
            # spectral parameters
            colr = gdat.stdvcolr[:gdat.numbener-1, None] * np.random.randn(gdat.numbcolr * gdat.numbstar).reshape((gdat.numbcolr, gdat.numbstar)).astype(np.float32) + \
                                                                                                                                            gdat.meancolr[:gdat.numbener-1, None]
            print 'colr'
            summgene(colr)
            print 'gdat.numbener'
            print gdat.numbener
            print 'fluxsumm'
            summgene(fluxsumm)
            print 'flux'
            summgene(flux)
            print 'colr[:, None, :]'
            summgene(colr[:, None, :])
            print 'flux[1:, :, :]'
            summgene(flux[1:, :, :])
            # temp
            for t in gdat.indxtime:
                print 'flux[1:, t, :]'
                summgene(flux[1:, t, :])
                flux[1:, t, :] *= 10**(0.4*colr)
        
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
    
            flux[:, :, :] *= difftemp[None, :, :]
            
            # inject transits
            indxstartran = np.random.choice(indxstar, size=gdat.numbstar/2, replace=False)
            for k in indxstartran:
                indxinit = np.random.choice(gdat.indxtime)
                indxtemp = np.arange(indxinit, indxinit + 4) % gdat.numblcpr
                flux[:, indxtemp, k] = np.random.rand()
        
            #flux[:, 1:, :] = fluxsumm[None, None, :] * lcpr[None, :, :]
        
        # evaluate model
        cntpdata = eval_modl(gdat, xpos, ypos, flux, trueback, numbsidepsfn, coefspix, clib=gdatnotp.clib.clib_eval_modl, sizeimag=sizeimag)
        
        if not np.isfinite(cntpdata).all():
            raise Exception('')
        
        cntpdata[cntpdata < 1] = 1.
        
        # add noise
        vari = cntpdata / gain
        cntpdata += (np.sqrt(vari) * np.random.normal(size=(gdat.numbener, sizeimag[1], sizeimag[0], gdat.numbtime))).astype(np.float32)
        
        if not np.isfinite(cntpdata).all():
            raise Exception('')
        
        # write to file
        path = pathdata + strgdata + '_%04d%04d_%s_mock.h5' % (gdat.numbener, gdat.numbtime, strgtype)
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


