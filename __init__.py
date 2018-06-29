# lets make this a package
import numpy as np

class gdatstrt(object):

    def __init__(self):
        pass
    
    def __setattr__(self, attr, valu):
        super(gdatstrt, self).__setattr__(attr, valu)


def summgene(varb):
   
    if len(varb) == 0 or varb is None or isinstance(varb, np.ndarray) and varb.size == 0:
        print varb
    else:
        print np.amin(varb)
        print np.amax(varb)
        print np.mean(varb)
        print varb.shape


