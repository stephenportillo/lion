import numpy as np

def psf_poly_fit(psf0, nbin):
        assert psf0.shape[0] == psf0.shape[1] # assert PSF is square
        npix = psf0.shape[0]

        # pad by one row and one column
        psf = np.zeros((npix+1, npix+1), dtype=np.float32)
        psf[0:npix, 0:npix] = psf0

        # make design matrix for each nbin x nbin region
        nc = npix/nbin # dimension of original psf
        nx = nbin+1
        y, x = np.mgrid[0:nx, 0:nx] / np.float32(nbin)
        x = x.flatten()
        y = y.flatten()
        A = np.array([np.full(nx*nx, 1, dtype=np.float32), x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y], dtype=np.float32).T
        # output array of coefficients
        cf = np.zeros((nc, nc, A.shape[1]), dtype=np.float32)

        # loop over original psf pixels and get fit coefficients
        for iy in xrange(nc):
         for ix in xrange(nc):
                # solve p = A cf for cf
                p = psf[iy*nbin:(iy+1)*nbin+1, ix*nbin:(ix+1)*nbin+1].flatten()
                AtAinv = np.linalg.inv(np.dot(A.T, A))
                ans = np.dot(AtAinv, np.dot(A.T, p))
                cf[iy,ix,:] = ans

        return cf

def image_model_eval(x, y, f, back, imsz, nc, cf, weights=None, ref=None, lib=None):
        assert x.dtype == np.float32
        assert y.dtype == np.float32
        assert f.dtype == np.float32
        assert cf.dtype == np.float32
        if ref is not None:
                assert ref.dtype == np.float32

        if weights is None:
                weights = np.full(imsz, 1., dtype=np.float32)

        nstar = x.size
        rad = nc/2 # 12 for nc = 25

        ix = np.ceil(x).astype(np.int32)
        dx = ix - x
        iy = np.ceil(y).astype(np.int32)
        dy = iy - y

        dd = np.stack((np.full(nstar, 1., dtype=np.float32), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, dx*dx*dy, dx*dy*dy, dy*dy*dy)).astype(np.float32) * f

        if lib is None:
                image = np.full((imsz[1]+2*rad+1,imsz[0]+2*rad+1), back, dtype=np.float32)
                recon = np.dot(dd.T, cf.T).reshape((nstar,nc,nc))
                for i in xrange(nstar):
                        image[iy[i]:iy[i]+rad+rad+1,ix[i]:ix[i]+rad+rad+1] += recon[i,:,:]

                image = image[rad:imsz[1]+rad,rad:imsz[0]+rad]

                if ref is not None:
                        diff = ref - image
                        diff2 = np.sum(diff*diff*weights)
        else:
                image = np.full((imsz[1], imsz[0]), back, dtype=np.float32)
                recon = np.zeros((nstar,nc*nc), dtype=np.float32)
                reftemp = ref
                if ref is None:
                        reftemp = np.zeros((imsz[1], imsz[0]), dtype=np.float32)
                diff2 = lib(imsz[0], imsz[1], nstar, nc, cf.shape[1], dd, cf, recon, ix, iy, image, reftemp, weights)

        if ref is not None:
                return image, diff2
	else:
		return image
