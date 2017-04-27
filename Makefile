default:
	icc -c -mkl -shared -static-intel -liomp5 -fPIC pcat-lion.c -o pcat-lion.o
	icc -shared -Wl,-soname,pcat-lion.so -o pcat-lion.so pcat-lion.o
