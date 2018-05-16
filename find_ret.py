import numpy as np
from scipy import optimize

def output_its(theta,ret):
    """
    Given theta and retartdance value, gives intensity.
    """
    its = (1 + (np.cos(2*theta))**2 + ((np.sin(2*theta))**2)*np.cos(ret))/2
    return its

err_func_ret = lambda p,its,theta: its - output_its(p[2]*theta+p[1],p[0])

p0 = [np.pi/2,0,1.1]

def find_retardance(mn,angles,p0=[np.pi/2,0,1.05]): # angles in degrees
    p1, success = optimize.leastsq(err_func_ret, p0[:], args=(mn/np.max(mn),angles*np.pi/180))
    return p1

### new
#err_func_ret = lambda p,its,theta: its - output_its(p[1]*theta,p[0])

#def find_retardance(mn,angles,p0=[np.pi/2,1.1]): # angles in degrees
#    imax = np.argmax(mn)
#    mn = np.roll(mn,-imax)
#    angles = angles - angles[imax]
#
#    while (np.sum((angles < 0)) > 0):
#        angles[angles<0] = angles[angles<0] + 2*np.pi
#
#    angles = np.roll(angles,-imax)
#
#    p1, success = optimize.leastsq(err_func_ret, p0[:], args=(mn/np.max(mn),angles*np.pi/180))
#    return p1


