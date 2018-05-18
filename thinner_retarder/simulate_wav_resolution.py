import numpy as np
from scipy import optimize


def gauss_kern(sigma,size=None):
    if size == None:
        size = int(sigma)
    #size = int(size)
    x = np.arange(-size,size+1)
    g = np.exp(-(x**2/sigma))
    g = g/np.sum(g)
    return g


def output_its(theta,ret):
    """
    Given theta and retartdance value, gives intensity.
    """
    its = (1 + (np.cos(2*theta))**2 + ((np.sin(2*theta))**2)*np.cos(ret))/2
    return its

err_func_ret = lambda p,its,theta: its - output_its(theta,p[0])

def find_retardance(mn,angles,p0=[np.pi/2]): # angles in rad
    p1, success = optimize.leastsq(err_func_ret, p0[:], args=(mn/np.max(mn),angles))
    return p1

def n_c(lam,A,B,C,D,F):
    lam = lam/1000
    n = A + B*lam**2/(lam**2 - C) + D*lam**2/(lam**2 -F)
    n = np.sqrt(n)
    return n


def no_c(lam):
    Ao = 1.73358749
    Bo = 0.96464345
    Co = 1.94325203 * 10**(-2)
    Do = 1.82831454
    Fo = 120
    no = n_c(lam,Ao,Bo,Co,Do,Fo)
    return no

def ne_c(lam):
    Ae = 1.35859695
    Be = 0.82427830
    Ce = 1.06689543*10**(-2)
    De = 0.14429128
    Fe = 120
    ne = n_c(lam,Ae,Be,Ce,De,Fe)
    return ne



#ne = n_c(ww/1000,Ae,Be,Ce,De,Fe)
#no = n_c(ww/1000,Ao,Bo,Co,Do,Fo)


retardance = lambda t,lam: (2*np.pi * t*10**6 * (no_c(lam) - ne_c(lam))/lam) % (np.pi)  # t in mm, lam in nm




lam = np.linspace(530,570,1000)
aa = np.linspace(0,2*np.pi,1000)

ret_in = retardance(2.5,lam)

ii = np.zeros([1000,1000])
ii_s = np.zeros([1000,1000])

kern_size=10
gs = gauss_kern(kern_size)

for i in range(len(aa)):         
    for j in range((len(lam))):                              
        r_tmp = retardance(2.5,lam[j])          
        #ret_in.append(r_tmp)
        ii[i,j] = output_its(aa[i],r_tmp)
    ii_s[i,:] = np.convolve(ii[i,:],gs)[kern_size:-kern_size]

ret_in = np.array(ret_in)
ret_out = []

for j in range(len(lam)):
    p1_ret = find_retardance(ii_s[:,j],aa)
    ret_out.append(p1_ret[0])

ret_out = np.array(ret_out)



def convolved_retardance(thick,res,lam):  #thickness in mm, res in nm, lam in nms 
    ret_in = retardance(thick,lam)
    aa = np.linspace(0,2*np.pi,1000)

    ii = np.zeros([1000,len(lam)])
    ii_s = np.zeros([1000,len(lam)])
    
    kern_size = 200#int(res) # in terms of number of pixels
    res_pixel = res/np.mean(np.diff(lam))
    gs = gauss_kern(res_pixel,kern_size)

    for i in range(len(aa)):         
        for j in range((len(lam))):                              
            r_tmp = ret_in[j]          
            #ret_in.append(r_tmp)
            ii[i,j] = output_its(aa[i],r_tmp)
        ii_s[i,:] = np.convolve(ii[i,:],gs)[kern_size:-kern_size]
    
    ret_in = np.array(ret_in)
    ret_out = []

    for j in range(len(lam)):
        p1_ret = find_retardance(ii_s[:,j],aa)
        ret_out.append(p1_ret[0])
    
    ret_out = np.array(ret_out)
    return ret_out


errfunc_ret_out = lambda p,lam, measured_ret: convolved_retardance(p[0],p[1],lam) - measured_ret

p0 = [1.2,2]

p1_thick,success = optimize.leastsq(errfunc_ret_out, p0[:], args=(a[:,0],a[:,1]*np.pi/180))


# Fitting at 45 degrees orientation

def i_45(thick,res,lam):
    ret_in = retardance(thick,lam)
    ii = np.zeros(len(lam))
    ii_s = np.zeros(len(lam))

    kern_size = 200#int(res) # in terms of number of pixels    
    res_pixel = res/np.mean(np.diff(lam))
    gs = gauss_kern(res_pixel,kern_size)

    for j in range((len(lam))):                              
        r_tmp = ret_in[j]          
        #ret_in.append(r_tmp)
        ii[j] = output_its(np.pi/4,r_tmp)
    ii_s = np.convolve(ii,gs)[kern_size:-kern_size]

    return ii_s

errfunc_45 = lambda p,lam, obs_i_45: i_45(p[0],p[1],lam) - obs_i_45

p0 = [1.2,2]

p1, success = optimize.leastsq(errfunc_45,p0[:], args=(ww,obs_i_45/np.max(obs_i_45)))

