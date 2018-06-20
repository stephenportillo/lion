from __init__ import *    
from pcat import psf_poly_fit, eval_modl
import numpy as np
import os

gdat = gdatstrt()

strgdata = 'sdsstimevari'
strgpsfn = 'sdss0921'
gdat.booltimebins = True
gdat.numbtime = 1

gdat.numblcpr = gdat.numbtime - 1

pathlion = os.environ['LION_PATH'] + '/'
pathdata = os.environ['LION_DATA_PATH'] + '/'
    
f = open(pathdata + strgpsfn + '_psfn.txt')
nc, nbin = [np.int32(i) for i in f.readline().split()]
f.close()
psf = np.loadtxt(pathdata + strgpsfn + '_psfn.txt', skiprows=1).astype(np.float32)
    
gdat.boolplotsave = False 
cf = psf_poly_fit(gdat, psf, nbin)
npar = cf.shape[0]


np.random.seed(0) # set seed to always get same catalogue
sizeimag = [300, 300] # image size width, height
gdat.numbstar = 9000
truex = (np.random.uniform(size=gdat.numbstar)*(sizeimag[0]-1)).astype(np.float32)
truey = (np.random.uniform(size=gdat.numbstar)*(sizeimag[1]-1)).astype(np.float32)
truealpha = np.float32(2.0)
trueminf = np.float32(250.)
truelogf = np.random.exponential(scale=1./(truealpha-1.), size=gdat.numbstar).astype(np.float32)
truef = trueminf * np.exp(truelogf)
trueback = np.float32(179.)
gain = np.float32(4.62)

noise = np.random.normal(size=(sizeimag[1],sizeimag[0])).astype(np.float32)
lcprtrue = 1e-6 * np.random.rand((gdat.numblcpr * gdat.numbstar)).reshape((gdat.numblcpr, gdat.numbstar)) + 1.

mock = eval_modl(gdat, truex, truey, truef, trueback, nc, cf, lcprtrue, sizeimag=sizeimag)
mock[mock < 1] = 1. # maybe some negative pixels
variance = mock / gain
mock += (np.sqrt(variance) * np.random.normal(size=(sizeimag[1],sizeimag[0]))).astype(np.float32)

f = open(pathdata + strgdata + '_pixl.txt', 'w')
f.write('%1d\t%1d\t1\n0.\t%0.3f' % (sizeimag[0], sizeimag[1], gain))
f.close()

np.savetxt(pathdata + strgdata + '_cntp.txt', mock)

np.savetxt(pathdata + strgdata + '_psfn.txt', psf, header='%1d\t%1d' % (nc, nbin), comments='')

truth = np.array([truex, truey, truef]).T
np.savetxt(pathdata + strgdata + '_true.txt', truth)



