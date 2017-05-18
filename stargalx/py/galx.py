import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set(color_codes=True)
import sys, os, h5py

def summgene(varb):
    print np.amin(varb)
    print np.amax(varb)
    print np.mean(varb)
    print varb.shape
    print


def retr_sers(numbsidegrid, sersindx):
    
    # approximation to the b_n factor
    fact = 1.9992 * sersindx - 0.3271

    # number of pixels out to which the profile is sampled
    numbsamp = 5.

    # generate the grid
    xpostemp = np.linspace(0., numbsamp, numbsidegrid)
    ypostemp = np.linspace(0., numbsamp, numbsidegrid)
    xpos, ypos = np.meshgrid(xpostemp, ypostemp)
    xpos = 2. * xpos - numbsamp
    ypos = 2. * ypos - numbsamp
    gridphon = np.vstack((xpos.flatten(), ypos.flatten()))

    # evaluate the Sersic profile
    amplphon = np.exp(-fact * np.sqrt(xpos**2 + ypos**2)**(1. / sersindx)).flatten()

    # normalize such that the sum is unity
    amplphon /= sum(amplphon)

    return gridphon, amplphon


def retr_tranphon(gridphon, amplphon, xposgalx, yposgalx, specgalx, argsfrst, argsseco, argsthrd, paratype='fris'):
    
    # find the transformation matrix if parametrization is 
    if paratype == 'fris':
        ## orientation vector of a frisbee
        avec = np.array([argsfrst, argsseco, argsthrd])
        normavec = np.sqrt(sum(avec**2))
        avecunit = avec / normavec
        bvec = np.array([argsseco, -argsfrst, 0.])
        normbvec = np.sqrt(sum(bvec**2))
        bvecunit = bvec / normbvec
        cvec = np.cross(avecunit, bvecunit)
        tranmatr = normavec * np.array([[argsseco / normbvec, cvec[0]], [-argsfrst / normbvec, cvec[1]]])
        cvecnorm = np.sqrt(argsfrst**2 * argsthrd**2 + argsseco**2 * argsthrd**2 + (argsfrst**2 + argsseco**2)**2)
        tranmatr = normavec * np.array([[argsseco / normbvec, argsfrst * argsthrd / cvecnorm], [-argsfrst / normbvec, argsseco * argsthrd / cvecnorm]])
        radigalx = normavec
    if paratype == 'sher':
        ## size, axis ratio, and orientation angle
        radigalx = argsfrst 
        shexgalx = argsseco 
        sheygalx = argsthrd 
        tranmatr = radigalx * np.array([[1., -shexgalx], [-sheygalx, 1.]])
    elif paratype == 'angl':
        ## size, horizontal and vertical shear
        radigalx = argsfrst 
        ratigalx = argsseco 
        anglgalx = argsthrd 
        tranmatr = radigalx * np.array([[ratigalx * np.cos(anglgalx), -np.sin(anglgalx)], [ratigalx * np.sin(anglgalx), np.cos(anglgalx)]])

    # transform the phonion grid
    gridphontran = np.matmul(tranmatr, gridphon)

    # positions of the phonions
    xposphon = xposgalx + gridphontran[0, :]
    yposphon = yposgalx + gridphontran[1, :]

    # spectrum of the phonions
    specphon = amplphon * specgalx * 7.2 * np.pi * radigalx**2
    
    return xposphon, yposphon, specphon


def retr_sizephon(specphon):
    
    # minimum and maximum marker sizes
    minm = 1e-1
    maxm = 1000.
    size = 30. * (np.log10(specphon) - np.log10(minm)) / (np.log10(maxm) - np.log10(minm))

    return size


def writ_truedata():
    
    pathlion = os.environ["LION_PATH"]
    sys.path.insert(0, pathlion)
    from image_eval import image_model_eval, psf_poly_fit
    
    dictglob = dict()

    pathliondata = os.environ["LION_DATA_PATH"] + '/data/'
    fileobjt = open(pathliondata + 'sdss.0921_psf.txt')
    numbsidepsfn, factsamp = [np.int32(i) for i in fileobjt.readline().split()]
    fileobjt.close()
    
    psfn = np.loadtxt(pathliondata + 'sdss.0921_psf.txt', skiprows=1).astype(np.float32)
    cpsf = psf_poly_fit(psfn, factsamp)
    cpsf = cpsf.reshape((-1, cpsf.shape[2]))
    
    np.random.seed(0)
    numbside = [100, 100]
    
    # generate stars
    numbstar = 100
    fluxdistslop = np.float32(2.0)
    minmflux = np.float32(250.)
    logtflux = np.random.exponential(scale=1. / (fluxdistslop - 1.), size=numbstar).astype(np.float32)
    
    dictglob['numbstar'] = numbstar
    dictglob['xposstar'] = (np.random.uniform(size=numbstar) * (numbside[0] - 1)).astype(np.float32)
    dictglob['yposstar'] = (np.random.uniform(size=numbstar) * (numbside[1] - 1)).astype(np.float32)
    dictglob['fluxdistslop'] = fluxdistslop
    dictglob['minmflux'] = minmflux
    dictglob['fluxstar'] = minmflux * np.exp(logtflux)
    dictglob['specstar'] = dictglob['fluxstar']
    
    dictglob['back'] = np.float32(179.)
    dictglob['gain'] = np.float32(4.62)
    
    # generate galaxies
    dictglob['numbgalx'] = 100
    
    dictglob['xposgalx'] = (np.random.uniform(size=numbstar) * (numbside[0] - 1)).astype(np.float32)
    dictglob['yposgalx'] = (np.random.uniform(size=numbstar) * (numbside[1] - 1)).astype(np.float32)
    dictglob['avecfrst'] = (0.2 * (np.random.uniform(size=numbstar) * - 0.5)).astype(np.float32)
    dictglob['avecseco'] = (np.random.uniform(size=numbstar) * 5.).astype(np.float32)
    dictglob['avecthrd'] = (np.random.uniform(size=numbstar) * 5.).astype(np.float32)
    dictglob['specgalx'] = dictglob['fluxstar'][None, :]
    
    sersindx = 4.
    numbsidegrid = 21
    gridphon, amplphon = retr_sers(numbsidegrid, sersindx)
    for k in range(dictglob['numbgalx']):
        xposphon, yposphon, specphon = retr_tranphon(gridphon, amplphon, dictglob['xposgalx'][k], dictglob['yposgalx'][k], dictglob['specgalx'][:, k], \
                                                                                        dictglob['avecfrst'][k], dictglob['avecseco'][k], dictglob['avecthrd'][k], 'fris')
    

    xposcand = np.concatenate((xposphon, dictglob['xposstar'])).astype(np.float32)
    yposcand = np.concatenate((yposphon, dictglob['yposstar'])).astype(np.float32)
    fluxcand = np.concatenate((specphon, dictglob['fluxstar'])).astype(np.float32)
    
    indx = np.where((xposcand < 100.) & (xposcand > 0.) & (yposcand < 100.) & (yposcand > 0.))[0]
    xposcand =  xposcand[indx]
    yposcand =  yposcand[indx]
    fluxcand =  fluxcand[indx]

    # generate data
    datacnts = image_model_eval(xposcand, yposcand, fluxcand, dictglob['back'], numbside, numbsidepsfn, cpsf)
    #datacnts[datacnts < 1] = 1. # maybe some negative pixels
    variance = datacnts / dictglob['gain']
    datacnts += (np.sqrt(variance) * np.random.normal(size=(numbside[1],numbside[0]))).astype(np.float32)
    dictglob['datacnts'] = datacnts 
    # auxiliary data
    dictglob['numbside'] = numbside
    dictglob['psfn'] = psfn
    
    # write data to disk
    path = pathliondata + 'true.h5'
    filearry = h5py.File(path, 'w')
    for attr, valu in dictglob.iteritems():
        filearry.create_dataset(attr, data=valu)
    filearry.close()
   

def main():
    
    # number of profile samples along the edge of the square grid
    numbsidegrid = 11

    # Sersic index
    sersindx = 4.

    # sample the Sersic profile to get the phonion positions and amplitudes
    gridphon, amplphon = retr_sers(numbsidegrid, sersindx)
    
    sizesqrt = 5. / np.sqrt(2.)
        
    # plot phonions whose positions have been stretched and rotated, and spectra rescaled
    if False:
        listparagalx = [ \
                        #[0.,   0.,  100.,        5.,        0.,              0., 'angl'], \
                        #[0.,  -2.,  100.,        5.,        0.,              0., 'angl'], \
                        #[-2., -2.,  100.,        5.,        0.,              0., 'angl'], \
                        #[-2.,  0.,  100.,        5.,        0.,              0., 'angl'], \
                        #
                        #[0.,   0.,  100.,        5.,        0.,              0., 'angl'], \
                        #[0.,   0.,  100.,       7.5,        0.,              0., 'angl'], \
                        #
                        #[0.,   0.,  100.,        5.,        0.,              0., 'angl'], \
                        #[0.,   0.,  500.,        5.,        0.,              0., 'angl'], \
                        #[0.,   0., 2500.,        5.,        0.,              0., 'angl'], \
                        #
                        #[0.,   0.,  100.,        5.,        1.,              0., 'angl'], \
                        #[0.,   0.,  100.,        5.,       1.2,              0., 'angl'], \
                        #[0.,   0.,  100.,        5.,       1.5,              0., 'angl'], \
                        #[0.,   0.,  100.,        5.,       1.5,      np.pi / 8., 'angl'], \
                        #[0.,   0.,  100.,        5.,       1.5,      np.pi / 4., 'angl'], \
                        #[0.,   0.,  100.,        5.,       1.5, 3. * np.pi / 8., 'angl'], \
                        #[0.,   0.,  100.,        5.,       1.5,      np.pi / 2., 'angl'], \
                        #
                        #[0.,   0.,  100.,        5.,        0.,              0., 'sher'], \
                        #[0.,   0.,  100.,        5.,        0.,             0.1, 'sher'], \
                        #[0.,   0.,  100.,        5.,        0.,             0.5, 'sher'], \
                        #[0.,   0.,  100.,        5.,        0.,              1., 'sher'], \
                        #[0.,   0.,  100.,        5.,        0.,              2., 'sher'], \
                        #[0.,   0.,  100.,        5.,       0.1,              0., 'sher'], \
                        #[0.,   0.,  100.,        5.,       0.5,              0., 'sher'], \
                        #[0.,   0.,  100.,        5.,        1.,              0., 'sher'], \
                        #[0.,   0.,  100.,        5.,        2.,              0., 'sher'], \
                        
                        [0.,   0.,  100.,        5.,        0.,              0., 'fris'], \
                        [0.,   0.,  100.,  sizesqrt,  sizesqrt,        sizesqrt, 'fris'], \
                        [0.,   0.,  100.,        5.,        5.,              5., 'fris'], \
                        [0.,   0.,  100.,        5.,        3.,              2., 'fris'], \
                        [0.,   0.,  100.,        5.,        1.,              2., 'fris'], \
                        [0.,   0.,  100.,        5.,        0.,              2., 'fris'], \
                        [0.,   0.,  100.,        5.,       -1.,              2., 'fris'], \
                       ] 
    else:
        listparagalx = []
        listavec = []
        numbiterfrst = 5
        numbiterseco = 5
        numbiterthrd = 5
        listavecfrst = np.linspace(0.1, 2., numbiterfrst)
        listavecseco = np.linspace(0.1, 2., numbiterseco)
        #listavecthrd = np.array([2.])#np.linspace(0.1, 2., numbiterthrd)
        listavecthrd = np.linspace(0.1, 2., numbiterthrd)
        for k in range(numbiterfrst):
            for l in range(numbiterseco):
                for m in range(numbiterthrd):
                    listparagalx.append([0., 0., 100., listavecfrst[k], listavecseco[l], listavecthrd[m], 'fris'])
                    listavec.append([listavecfrst[k], listavecseco[l], listavecthrd[m]])
    
    plot_phon(listparagalx, gridphon, amplphon, listavec)


def plot_phon(listparagalx, gridphon, amplphon, listavec=None):
    
    # minimum flux allowed by the metamodel in ADU
    minmflux = 250

    pathplot = os.environ["LION_DATA_PATH"] + '/imag/'
    os.system('mkdir -p ' + pathplot)
    
    os.system('rm -rf ' + pathplot + '*')
    numbiter = len(listparagalx)

    for k in range(numbiter):
        
        xposgalx, yposgalx, specgalx, argsfrst, argsseco, argsthrd, paratype = listparagalx[k]
        figr, axis = plt.subplots(figsize=(6, 6))
        
        xposphon, yposphon, specphon = retr_tranphon(gridphon, amplphon, xposgalx, yposgalx, specgalx, argsfrst, argsseco, argsthrd, paratype)
        size = retr_sizephon(specphon)
        
        indxgrtr = np.where(specphon > minmflux)
        axis.scatter(xposphon, yposphon, s=size, color='g')
        axis.scatter(xposphon[indxgrtr], yposphon[indxgrtr], s=size[indxgrtr], color='b')
        
        if listavec != None:
            axis.set_title('$A_x=%4.2g A_y=%4.2g A_z=%4.2g$' % (listavec[k][0], listavec[k][1], listavec[k][2]))
        axis.set_xlim([-50, 50])
        axis.set_ylim([-50, 50])
        axis.set_xlabel('$x$')
        axis.set_xlabel('$y$')
    
        path = pathplot + 'galx%s.pdf' % k
        figr.savefig(path)
        plt.close(figr)


#main()
writ_truedata()
