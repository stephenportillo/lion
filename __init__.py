# lets make this a package
import numpy as np

def summgene(varb):
    
    if varb == None:
        print varb
    else:
        print np.amin(varb)
        print np.amax(varb)
        print np.mean(varb)
        print varb.shape


