import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set(color_codes=True)
import os

def retr_sers(numbsidegrid, sersindx):
    
    # approximation to the b_n factor
    fact = 1.9992 * sersindx - 0.3271

    # number of pixels out to which the profile is sampled
    numbsamp = 5.

    # generate the grid
    xaxitemp = np.linspace(0., numbsamp, numbsidegrid)
    yaxitemp = np.linspace(0., numbsamp, numbsidegrid)
    xaxi, yaxi = np.meshgrid(xaxitemp, yaxitemp)
    xaxi = 2. * xaxi - numbsamp
    yaxi = 2. * yaxi - numbsamp
    gridphon = np.vstack((xaxi.flatten(), yaxi.flatten()))

    # evaluate the Sersic profile
    amplphon = np.exp(-fact * np.sqrt(xaxi**2 + yaxi**2)**(1. / sersindx)).flatten()

    # normalize such that the sum is unity
    amplphon /= sum(amplphon)

    return gridphon, amplphon


def retr_tranphon(gridphon, amplphon, xaxigalx, yaxigalx, specgalx, radigalx, ratigalx, anglgalx):
    
    # find the transformation matrix for the given half light radius, axis ratio and orientation angle
    tranmatr = radigalx * np.array([[ratigalx * np.cos(anglgalx), -np.sin(anglgalx)], [ratigalx * np.sin(anglgalx), np.cos(anglgalx)]])

    # transform the phonion grid
    gridphontran = np.matmul(tranmatr, gridphon)

    # positions of the phonions
    xaxiphon = xaxigalx + gridphontran[0, :]
    yaxiphon = yaxigalx + gridphontran[1, :]

    # spectrum of the phonions
    specphon = amplphon * specgalx * 7.2 * np.pi * radigalx**2
    
    return xaxiphon, yaxiphon, specphon


def retr_sizephon(specphon):
    
    # minimum and maximum marker sizes
    minm = 1e-1
    maxm = 1000.
    size = 30. * (np.log10(specphon) - np.log10(minm)) / (np.log10(maxm) - np.log10(minm))

    return size


def main():
    
    # number of profile samples along the edge of the square grid
    numbsidegrid = 11

    # Sersic index
    sersindx = 4.

    # sample the Sersic profile to get the phonion position and flux grids
    gridphon, amplphon = retr_sers(numbsidegrid, sersindx)
    
    # plot phonions whose positions have been stretched and rotated, and spectra rescaled
    listparagalx = [ \
                    [0.,   0.,  100.,  5.,  1.,              0.], \
                    [0.,  -2.,  100.,  5.,  1.,              0.], \
                    [-2., -2.,  100.,  5.,  1.,              0.], \
                    [-2.,  0.,  100.,  5.,  1.,              0.], \
                    
                    [0.,   0.,  100.,  5.,  1.,              0.], \
                    [0.,   0.,  100., 7.5,  1.,              0.], \
                    
                    [0.,   0.,  100.,  5.,  1.,              0.], \
                    [0.,   0.,  500.,  5.,  1.,              0.], \
                    [0.,   0., 2500.,  5.,  1.,              0.], \
                    
                    [0.,   0.,  100.,  5.,  1.,              0.], \
                    [0.,   0.,  100.,  5., 1.2,              0.], \
                    [0.,   0.,  100.,  5., 1.5,              0.], \
                    [0.,   0.,  100.,  5., 1.5,      np.pi / 8.], \
                    [0.,   0.,  100.,  5., 1.5,      np.pi / 4.], \
                    [0.,   0.,  100.,  5., 1.5, 3. * np.pi / 8.], \
                    [0.,   0.,  100.,  5., 1.5,      np.pi / 2.], \
                   ] 
    plot_phon(listparagalx, gridphon, amplphon)


def plot_phon(listparagalx, gridphon, amplphon):
    
    # minimum flux allowed by the metamodel in ADU
    minmflux = 250

    pathplot = os.environ["TDGU_DATA_PATH"] + '/stargalx/imag/'
    os.system('mkdir -p ' + pathplot)
    
    numbiter = len(listparagalx)

    for k in range(numbiter):
        
        xaxigalx, yaxigalx, specgalx, radigalx, ratigalx, anglgalx = listparagalx[k]
        figr, axis = plt.subplots(figsize=(6, 6))
        
        xaxiphon, yaxiphon, specphon = retr_tranphon(gridphon, amplphon, xaxigalx, yaxigalx, specgalx, radigalx, ratigalx, anglgalx)
        size = retr_sizephon(specphon)
        
        indxgrtr = np.where(specphon > minmflux)
        axis.scatter(xaxiphon, yaxiphon, s=size, color='g')
        axis.scatter(xaxiphon[indxgrtr], yaxiphon[indxgrtr], s=size[indxgrtr], color='b')
        
        axis.set_xlim([-50, 50])
        axis.set_ylim([-50, 50])
        axis.set_xlabel('$x$')
        axis.set_xlabel('$y$')
    
        path = pathplot + 'galx%s.pdf' % k
        figr.savefig(path)
        plt.close(figr)


main()

