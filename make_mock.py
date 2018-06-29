from __init__ import *    
from pcat import psf_poly_fit, eval_modl
from pcat import *
import numpy as np
import os

gdat = gdatstrt()

strgdata = 'sdsstimevari'
strgpsfn = 'sdss0921'
gdat.booltimebins = True
gdat.numbtime = 20
indxtime = np.arange(gdat.numbtime)
gdat.numblcpr = gdat.numbtime - 1
gdat.pathlion = os.environ['LION_PATH'] + '/'
pathdata = os.environ['LION_DATA_PATH'] + '/'
    
# setup
gdat.boolplotsave = False 
setp(gdat)
gdat.verbtype = 1

f = open(pathdata + strgpsfn + '_psfn.txt')
nc, nbin = [np.int32(i) for i in f.readline().split()]
f.close()
psf = np.loadtxt(pathdata + strgpsfn + '_psfn.txt', skiprows=1).astype(np.float32)
    
coefspix = psf_poly_fit(gdat, psf, nbin)

np.random.seed(1) # set seed to always get same catalogue
sizeimag = [300, 300] # image size width, height
gdat.numbstar = 9000
indxstar = np.arange(gdat.numbstar)

# background
trueback = np.float32(179.)

# gain
gain = np.float32(4.62)

# position
xpos = (np.random.uniform(size=gdat.numbstar)*(sizeimag[0]-1)).astype(np.float32)
ypos = (np.random.uniform(size=gdat.numbstar)*(sizeimag[1]-1)).astype(np.float32)

# flux
trueslop = np.float32(2.0)
trueminf = np.float32(250.)
truelogf = np.random.exponential(scale=1./(trueslop-1.), size=gdat.numbstar).astype(np.float32)
flux = trueminf * np.exp(truelogf)

# temporal parameters
print 'gdat.numblcpr * gdat.numbstar'
print gdat.numblcpr * gdat.numbstar
lcpr = 1e-6 * np.random.randn(gdat.numblcpr * gdat.numbstar)

print 'lcpr'
print lcpr
summgene(lcpr)
lcpr = 1e-6 * np.random.randn((gdat.numblcpr * gdat.numbstar)).reshape((gdat.numblcpr, gdat.numbstar)).astype(np.float32) + 1.

# inject transits
indxstartran = np.random.choice(indxstar, size=gdat.numbstar/2, replace=False)
print 'lcpr'
summgene(lcpr)
for k in indxstartran:
    indxinit = np.random.choice(indxtime)
    print 'k'
    print k
    print 'indxinit'
    print indxinit
    print
    lcpr[indxinit:indxinit+4, k] = 0.

for k in range(10):
    print 'k'
    print k
    print 'lcpr[:, k]'
    print lcpr[:, k]
    print

print 'lcpr'
summgene(lcpr)
print 'xpos'
summgene(xpos)
print 'ypos'
summgene(ypos)
print 'flux'
summgene(flux)
# evaluate model
cntpdata = eval_modl(gdat, xpos, ypos, flux, trueback, nc, coefspix, lcpr, lib=gdat.libmmult.pcat_model_eval, sizeimag=sizeimag)

cntpdata = cntpdata[:, :, :, None]

if not np.isfinite(cntpdata).all():
    raise Exception('')

# temp -- why negative?
cntpdata[cntpdata < 1] = 1.

print 'cntpdata'
summgene(cntpdata)

# add noise
vari = cntpdata / gain
print 'vari'
summgene(vari)
cntpdata += (np.sqrt(vari) * np.random.normal(size=(gdat.numbtime, sizeimag[1], sizeimag[0], 1))).astype(np.float32)

print 'cntpdata'
summgene(cntpdata)
if not np.isfinite(cntpdata).all():
    raise Exception('')

# write to file
path = pathdata + strgdata + '_mock.h5'
print 'Writing to %s...' % path
filetemp = h5py.File(path, 'w')
filetemp.create_dataset('cntpdata', data=cntpdata)
filetemp.create_dataset('numb', data=gdat.numbstar)
filetemp.create_dataset('xpos', data=xpos)
filetemp.create_dataset('ypos', data=ypos)
filetemp.create_dataset('flux', data=flux)
filetemp.create_dataset('gain', data=gain)
if gdat.booltimebins:
    filetemp.create_dataset('lcpr', data=lcpr)
filetemp.close()



