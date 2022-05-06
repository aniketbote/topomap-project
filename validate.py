from ripser import Rips
from numpy import inf
import numpy as np

def validate_points(a, b, eps = 1e-4, verbose = 0):
    rips = Rips(verbose = False)
    d1 = rips.fit_transform(a)
    rips = Rips(verbose = False)
    d2 = rips.fit_transform(b)
    h0_o = d1[0]
    h0_t = d2[0]
    if h0_o.shape != h0_t.shape:
        if verbose:
            print('Different number of pairs present for Rips Filtration')
        return False
    h0_o[h0_o == inf] = 999999999
    h0_t[h0_t == inf] = 999999999
    mask = np.abs(h0_o - h0_t) > eps
    if np.sum(mask) == 0:
        return True
    else:
        if verbose:
            for c, (i,j) in enumerate(mask):
                if i == True or j == True:
                    print(f"Original H0: ({h0_o[c][0]:4.4f},{h0_o[c][1]:4.4f})  Transformed H0: ({h0_t[c][0]:4.4f},{h0_t[c][1]:4.4f})")
        return False
       