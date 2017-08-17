import numpy as np

def retr_factsers(sersindx):
    factsers = 1.9992 * sersindx - 0.3271
    return factsers

def retr_sers(numbsidegrid=20, sersindx=4., factusam=100, factradigalx=2., pathplot=None):
    # numbsidegrid is the number of pixels along one side of the grid and should be an odd number
    numbsidegridusam = numbsidegrid * factusam

    # generate the grid
    xpostemp = np.linspace(-factradigalx, factradigalx, numbsidegridusam + 1)
    ypostemp = np.linspace(-factradigalx, factradigalx, numbsidegridusam + 1)
    xpos, ypos = np.meshgrid(xpostemp, ypostemp)

    # evaluate the Sersic profile
    ## approximation to the b_n factor
    factsers = retr_factsers(sersindx)
    ## profile sampled over the square grid
    amplphonusam = np.exp(-factsers * (np.sqrt(xpos**2 + ypos**2)**(1. / sersindx) - 1.))#.flatten()
    ## down sample
    amplphon = np.empty((numbsidegrid + 1, numbsidegrid + 1))
    for k in range(numbsidegrid + 1):
        for l in range(numbsidegrid + 1):
            amplphon[k, l] = np.mean(amplphonusam[k*factusam:(k+1)*factusam, l*factusam:(l+1)*factusam])
    indx = np.linspace(factusam / 2, numbsidegridusam - factusam / 2 + 1, numbsidegrid + 1, dtype=int)

    xpostemp = xpostemp[indx]
    ypostemp = ypostemp[indx]

    xpos, ypos = np.meshgrid(xpostemp, ypostemp)

    amplphon = amplphon.flatten()
    ## normalize such that the sum is unity
    amplphon /= sum(amplphon)

    gridphon = np.vstack((xpos.flatten(), ypos.flatten()))

    return gridphon, amplphon

def retr_tranphon(gridphon, amplphon, xposgalx, yposgalx, fluxgalx, argsfrst, argsseco, argsthrd):
    assert (argsfrst+argsseco+argsthrd != 0).all()
    ## orientation vector of a frisbee
    avec = np.column_stack([argsfrst, argsseco, argsthrd])
    normavec = np.sqrt(np.sum(avec*avec, axis=1))
    #avecunit = avec / normavec
    bvec = np.column_stack([argsseco, -argsfrst])
    normbvec = np.sqrt(np.sum(bvec*bvec, axis=1))
    #bvecunit = bvec / normbvec
    #cvec = np.cross(avecunit, bvecunit)
    #tranmatr = normavec * np.array([[argsseco / normbvec, cvec[0]], [-argsfrst / normbvec, cvec[1]]])
    #cvecnorm = np.sqrt(argsfrst**2 * argsthrd**2 + argsseco**2 * argsthrd**2 + (argsfrst**2 + argsseco**2)**2)
    cvecnorm = np.sqrt(normbvec*normbvec*(argsthrd*argsthrd + normbvec*normbvec))

    tranmatr = np.empty((xposgalx.size, 2, 2))
    tranmatr[:,0,0] = argsseco / normbvec
    tranmatr[:,0,1] = argsfrst * argsthrd / cvecnorm
    tranmatr[:,1,0] = -argsfrst / normbvec
    tranmatr[:,1,1] = argsseco * argsthrd / cvecnorm

    # transform the phonion grid
    gridphontran = np.matmul(tranmatr, gridphon)

    # positions of the phonions
    xposphon = xposgalx[:, None] + gridphontran[:, 0, :]
    yposphon = yposgalx[:, None] + gridphontran[:, 1, :]

    # spectrum of the phonions
    specphon = fluxgalx[:, None] * amplphon[None, :]

    return (xposphon.flatten()).astype(np.float32), (yposphon.flatten()).astype(np.float32), (specphon.flatten()).astype(np.float32)

#gridphon, amplphon = retr_sers()
#n = 10
#x = np.random.normal(size=n)
#y = np.random.normal(size=n)
#z = np.random.normal(size=n)
#xphon, yphon, sphon = retr_tranphon(gridphon, amplphon, x, y, z, x, y, z)
#print xphon.shape, yphon.shape, sphon.shape
