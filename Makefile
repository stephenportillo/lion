default:
	icc -mkl -shared -static-intel -liomp5 -fPIC pcat-lion.c -o pcat-lion.so
