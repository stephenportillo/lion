#include <stdlib.h>
#include <stdbool.h>
#include "mkl_cblas.h"
#include "i_malloc.h"
#define max(a,b) ({ typeof (a) _a = (a); typeof (b) _b = (b); _a > _b ? _a : _b; })
#define min(a,b) ({ typeof (a) _a = (a); typeof (b) _b = (b); _a < _b ? _a : _b; })

void pcat_imag_acpt(int NX, int NY, float* image, float* image_acpt, int* reg_acpt, int regsize, int margin, int offsetx, int offsety){
    int NREGX = (NX / regsize) + 1;
    int NREGY = (NY / regsize) + 1;
    int y0, y1, x0, x1, i, j, ii, jj;
    for (j=0 ; j < NREGY ; j++){
        y0 = max(j*regsize-offsety-margin, 0);
        y1 = min((j+1)*regsize-offsety+margin, NY);
        for (i=0 ; i < NREGX ; i++){
                x0 = max(i*regsize-offsetx-margin, 0);
                x1 = min((i+1)*regsize-offsetx+margin, NX);
                if (reg_acpt[j*NREGX+i] > 0){
                    for (jj=y0 ; jj<y1; jj++)
                     for (ii=x0 ; ii<x1; ii++)
                        image_acpt[jj*NX+ii] = image[jj*NX+ii];
                }
        }
    }
}

void pcat_like_eval(int NX, int NY, float* image, float* ref, float* weight, double* diff2, int regsize, int margin, int offsetx, int offsety){
    int NREGX = (NX / regsize) + 1;
    int NREGY = (NY / regsize) + 1;
    int y0, y1, x0, x1, i, j, ii, jj;
    for (j=0 ; j < NREGY ; j++){
        y0 = max(j*regsize-offsety-margin, 0);
        y1 = min((j+1)*regsize-offsety+margin, NY);
        for (i=0 ; i < NREGX ; i++){
                x0 = max(i*regsize-offsetx-margin, 0);
                x1 = min((i+1)*regsize-offsetx+margin, NX);
                diff2[j*NREGX+i] = 0.;
                for (jj=y0 ; jj<y1; jj++)
                 for (ii=x0 ; ii<x1; ii++)
                    diff2[j*NREGX+i] += (image[jj*NX+ii]-ref[jj*NX+ii])*(image[jj*NX+ii]-ref[jj*NX+ii]) * weight[jj*NX+ii];
        }
    }
}

void pcat_model_eval(int NX, int NY, int numbphon, int nc, int k, float* A, float* B, float* C, int* x,
	                 int* y, float* image, float* ref, float* weight, double* diff2, int regsize, int margin, int offsetx, int offsety,
                     int numbtime, int booltimebins, float* lcpr)
{
    int i, m, t, imax, j, r, jmax, rad, p, xposthis, yposthis;
    float alpha, beta;
    int n = nc*nc;
    rad = nc / 2;
    alpha = 1.; beta = 0.;

    // save time if there are many phonions per pixel by overwriting and shorting the A matrix
    int hash[NY*NX];
    for (i=0; i<NY*NX; i++)
        hash[i] = -1;
    int numbphonshrt = 0;
    for (p = 0; p < numbphon; p++){
        xposthis = x[p];
        yposthis = y[p];
        int idx = yposthis*NX+xposthis;
        if (hash[idx] != -1){
            for (i=0; i<k; i++){
                A[hash[idx]*k+i] += A[p*k+i];
            }
        }
        else{
            hash[idx] = numbphonshrt;
            for (i=0; i<k; i++)
                A[numbphonshrt*k+i] = A[p*k+i];
            x[numbphonshrt] = x[p];
            y[numbphonshrt] = y[p];
            numbphonshrt++;
        }
    }
    numbphon = numbphonshrt;

    //  matrix multiplication
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, numbphon, n, k, alpha, A, k, B, n, beta, C, n);

    //  loop over phonions, insert psfs into image    
    for (p = 0 ; p < numbphon ; p++){
	    xposthis = x[p];
	    yposthis = y[p];
	    imax = min(xposthis+rad, NX-1);
	    jmax = min(yposthis+rad, NY-1);
	    for (j = max(yposthis-rad, 0), r = (p*nc+j-yposthis+rad)*nc ; j <= jmax ; j++, r+=nc){
	        for (i = max(xposthis - rad, 0), m = i-xposthis+rad ; i <= imax ; i++, m++){
		        if (booltimebins > 0){
                    for (t = 0; t < numbtime; t++){
                        if (t == 0){
                            image[t, j*NX+i] += C[m+r];
                        }
                        else{
                            image[t, j*NX+i] += C[m+r] * lcpr[t-1];
                        }
                    }
                }
                else{
                    image[j*NX+i] += C[m+r];
                }
            }
        }
    }

    pcat_like_eval(NX, NY, image, ref, weight, diff2, regsize, margin, offsetx, offsety);
}
