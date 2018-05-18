from astropy.io import fits
import cv2
from scipy import optimize
import numpy as np

#rotation
rows,cols = img.shape 
M = cv2.getRotationMatrix2D((cols/2,rows/2),1.4,1)

dst = cv2.warpAffine(img,M,(cols,rows))

mn = np.mean(dst,axis=0)

xid = np.array(range(len(mn)))


#reading filters
fl = '/home/abhilash/lctf/fixed_ret/broad_filters/550FS10.csv'
flt_dt = np.loadtxt(fl,skiprows=9,delimiter=',')
wl = flt_dt[:,0]
tx = flt_dt[:,1]

wl_i = np.linspace(wl[0],wl[-1],len(mn))
tx_i = np.interp(wl_i,wl,tx)


id2wav = lambda a,b,xid: a*xid + b

errfunc = lambda p,i,i_csv,lam: np.interp(lam,id2wav(p[0],p[1],np.arange(len(i))),i) - i_csv 
#errfunc = lambda 


mn = mn/np.max(mn)
tx = tx/np.max(tx)
p0 = [1/45,520]
p1, success = optimize.leastsq(errfunc, p0[:], args=(mn,tx,wl))

