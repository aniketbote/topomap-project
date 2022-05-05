import numpy as np
def find_angle(p1, p2, trans):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    n = np.sqrt(x*x + y*y)
    x = x/n
    y = y/n
    trans['cos'] = x
    trans['sin'] = np.sqrt(1 - x*x)
    
    if y >= 0:
        trans['sin'] = -trans['sin']
        
    return trans
    
    
    