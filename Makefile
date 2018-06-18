default:
	#icc -mkl -shared -static-intel -liomp5 -fPIC -O2 blas.c -o blas.so
	gcc -mavx -shared -fPIC -O2 blas.c -o blas.so
