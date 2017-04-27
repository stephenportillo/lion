/*
 * icc -c -mkl -shared -static-intel -liomp5 -fPIC pcat-lion.c mklsmall.so -o pcat-lion.o
 * icc -shared -Wl,-soname,pcat-lion.so -o pcat-lion.so pcat-lion.o
 */
#include <stdlib.h>
#include "mkl_cblas.h"
#include "i_malloc.h"
#define max(a,b) \
    ({ typeof (a) _a = (a);    \
	typeof (b) _b = (b);   \
        _a > _b ? _a : _b; })
#define min(a,b) \
    ({ typeof (a) _a = (a);    \
	typeof (b) _b = (b);   \
        _a < _b ? _a : _b; })
double pcat_model_eval(int NX, int NY, int nstar, int nc, int k, float* A, float* B, float* C, int* x,
	int* y, float* image, float* ref, float* weight)
{
    int      i,i2,imax,j,j2,jmax,rad,istar,xx,yy;
    float    alpha, beta;

    int n = nc*nc;
    rad = nc/2;

    alpha = 1.0; beta = 0.0;

//  matrix multiplication
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
        nstar, n, k, alpha, A, nstar, B, k, beta, C, n);

//  loop over stars, insert psfs into image    
    for (istar = 0 ; istar < nstar ; istar++)
    {
	xx = x[istar];
	yy = y[istar];
	imax = min(xx+rad,NX-1);
	jmax = min(yy+rad,NY-1);
	for (j = max(yy-rad,0), j2 = (istar*nc+j-yy+rad)*nc ; j <= jmax ; j++, j2+=nc)
	    for (i = max(xx-rad,0), i2 = i-xx+rad ; i <= imax ; i++, i2++)
		image[j*NX+i] += C[i2+j2];
    }

    double diff2 = 0.0;
    for (j=0 ; j < NY ; j++)
	for (i=0 ; i < NX ; i++)
		diff2 += (image[j*NX+i]-ref[j*NX+i])*(image[j*NX+i]-ref[j*NX+i]) * weight[j*NX+i];

    return diff2;
}
