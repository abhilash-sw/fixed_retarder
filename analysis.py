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
fl = '/media/abhilash/Deep Thought/tmp/filters/550FS10.csv' #/home/abhilash/lctf/fixed_ret/broad_filters/550FS10.csv'
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
p0 = [-1/46,578] #p0 = [1/45,520] the wavelength axis is in opposite direction
#p1, success = optimize.leastsq(errfunc, p0[:], args=(mn,tx,wl))


mn_sm = lowess(mn,xid,frac=0.06)
mn_sm = mn_sm[:,1]
p1, success = optimize.leastsq(errfunc, p0[:], args=(mn_sm,tx,wl))






# find tilt in the image
cf = []
cor = []
rm = []

angs = np.arange(1,2,0.001)

for ang in np.arange(1,2,0.001):
    M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    r1 = np.mean(dst[450:550,:],axis=0)
    r2 = np.mean(dst[1450:1550,:],axis=0)
    cor.append(np.correlate(r1,r2)[0])
    rm.append(np.var(r1-r2)) 
    cf.append(np.corrcoef(r1,r2)[0,1])

imax = np.argmax(cf)

M  = cv2.getRotationMatrix2D((cols/2,rows/2),angs[imax],1)








#
import glob

img_files = glob.glob('*-.fits')
mns = []
angles = []

for img_file in img_files:
    print(img_file)
    angles.append(float(img_file[:-6]))
    hdu = fits.open(img_file)
    img = hdu[0].data
    hdu.close()
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst = dst[350:1850,:]
    mn = np.mean(dst,axis=0)
    mns.append(np.mean(mn[1248:1298]))

mns = np.array(mns)
angles = np.array(angles)

ind = np.argsort(angles)
angles = angles[ind]
mns = mns[ind]


################# NEW ###################
from astropy.io import fits
import cv2
from scipy import optimize
import numpy as np
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

hdus = fits.open('45.0-.fits')
img = hdus[0].data
hdus.close()
#img = np.fliplr(img)
rows,cols = img.shape

# find tilt in the image
cf = []
cor = []
rm = []

angs = np.arange(1,2,0.001)

for ang in np.arange(1,2,0.001):
    M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    r1 = np.mean(dst[450:550,:],axis=0)
    r2 = np.mean(dst[1450:1550,:],axis=0)
    cor.append(np.correlate(r1,r2)[0])
    rm.append(np.var(r1-r2)) 
    cf.append(np.corrcoef(r1,r2)[0,1])

imax = np.argmax(cf)

M  = cv2.getRotationMatrix2D((cols/2,rows/2),angs[imax],1)

dst = cv2.warpAffine(img,M,(cols,rows))

dst = np.fliplr(dst)

mn = np.mean(dst,axis=0)

xid = np.array(range(len(mn)))


# filter reading and fit
fl = '/home/abhilash/lctf/fixed_ret/broad_filters/550FS10.csv'


flt_dt = np.loadtxt(fl,skiprows=9,delimiter=',')
wl = flt_dt[:,0]
tx = flt_dt[:,1]

#wl_i = np.linspace(wl[0],wl[-1],len(mn))
#tx_i = np.interp(wl_i,wl,tx)


id2wav = lambda a,b,xid: a*xid + b
wav2id = lambda a,b,lam: (lam-b)/a

errfunc = lambda p,i,i_csv,lam: np.interp(lam,id2wav(p[0],p[1],np.arange(len(i))),i) - i_csv
#errfunc = lambda


mn = mn/np.max(mn)
tx = tx/np.max(tx)
p0 = [-1/46,578] #p0 = [1/45,520] the wavelength axis is in opposite direction
#p1, success = optimize.leastsq(errfunc, p0[:], args=(mn,tx,wl))


mn_sm = lowess(mn,xid,frac=0.06)
mn_sm = mn_sm[:,1]
p1, success = optimize.leastsq(errfunc, p0[:], args=(mn_sm,tx,wl))

