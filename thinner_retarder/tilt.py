import numpy as np
from smooth_numpy import smooth
from scipy.signal import argrelextrema
from astropy.io import fits


rot_angs = []
angles = []
for img_file in img_files:                                            
    hdus = fits.open(img_file)                        
    print(img_file)                        
    img = hdus[0].data                 
    hdus.close()                         
    rows,cols = img.shape             
                             
# find tilt in the image              
    cf = []
    #cor = []
    #rm = []

    angs = np.arange(1,2,0.001)

    for ang in np.arange(1,2,0.001):
        M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        r1 = np.mean(dst[450:550,:],axis=0)
        r2 = np.mean(dst[1450:1550,:],axis=0)
        #cor.append(np.correlate(r1,r2)[0])
        #rm.append(np.var(r1-r2)) 
        cf.append(np.corrcoef(r1,r2)[0,1])
    imax = np.argmax(cf)
    rot_angs.append(angs[imax]) 
    angles.append(float(img_file[:-6]))       


tmp = rot_angs
rot_angs = {}

for i in range(len(tmp)):
    rot_angs[angles[i]] = tmp[i]




wav_ids = {}
intensities = {}


xinit = np.arange(1100,1500)#np.arange(500,2000)
yinit = 500*np.ones(len(xinit))

y = np.arange(500,1000)

#theta = 0

for img_file in img_files:
    print(img_file)
    hdus = fits.open(img_file)
    img = hdus[0].data
    hdus.close()

    theta = float(img_file[:-6])
    m = np.tan(-(90 - rot_angs[theta])*np.pi/180)

    c = yinit - xinit*m

    x = np.zeros([len(y),len(c)])
    tilt_img =  np.zeros([len(y),len(c)])

    for i in range(len(c)):
        x[:,i] = (y-c[i])/m
        tilt_img[:,i] = img[y.astype('int'),x[:,i].astype('int')]

    tilt_mn = np.mean(tilt_img,axis=0)

    tilt_mn_s = smooth(tilt_mn)
    tilt_mn_s = tilt_mn_s[5:-5]

    imaxes = argrelextrema(tilt_mn_s, np.greater)[0]

    wav_ids[theta] = imaxes + xinit[0]
    intensities[theta] = tilt_mn_s[imaxes]




aaa = []
fpeak = []

for k in intensities.keys():
    if len(intensities[k]) == 9:
        aaa.append(k)             
        fpeak.append(intensities[k][0])

aaa = np.array(aaa)
fpeak = np.array(fpeak)

ind = np.argsort(aaa)
aaa = aaa[ind]
fpeak = fpeak[ind]
