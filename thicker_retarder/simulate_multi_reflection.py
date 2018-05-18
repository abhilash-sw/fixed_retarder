import numpy as np
from scipy import optimize

def output_its(theta,ret):
    """
    Given theta and retartdance value, gives intensity. theta and ret in radians
    """
    its = (1 + (np.cos(2*theta))**2 + ((np.sin(2*theta))**2)*np.cos(ret))/2
    return its

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

retardance = lambda thick,lam: (2*np.pi * thick*10**6 * (no_c(lam/1000) - ne_c(lam/1000))/lam) % (np.pi)  # t in mm, lam in nm

def transmittance(n,thick,lam): #n - ri,thick in mm, lam in nm
    delta = 2*np.pi * thick * 10**6 * n/lam
    t = 8*n**2/(1 + n**4 + 6*n**2 - (n**2 -1)**2*np.cos(2*delta))
    return t

def modified_retardance(n,thick,lam): #n - ri,thick in mm, lam in nm
    delta = 2*np.pi * thick * 10**6 * n/lam
    tan_phi = (n**2 + 1)/(2*n) * np.tan(delta)
    phi = np.arctan(tan_phi)
    return phi


def modified_output_its(theta,lam,thick): #thick in mm, lam in nm
    ne = ne_c(lam)
    no = no_c(lam)
    phi_e = modified_retardance(ne,thick,lam)
    phi_o = modified_retardance(no,thick,lam)
    phi = phi_e - phi_o

    te = transmittance(ne,thick,lam)
    to = transmittance(no,thick,lam)
    its_tmp = te*np.cos(theta)**4 + to*np.sin(theta)**4 + 2*(te*to)**(1/2)*np.sin(theta)**2*np.cos(theta)**2*np.cos(phi)
    its =  its_tmp 
    return its

start_id = 1000
end_id = 1500
step_id =1 

ret = {}

obs_its_wrt_angles = []
lams = []
modified_angles = []


for ids in np.arange(start_id,end_id,step_id):
    mn_id = []
    for ds in angles:
        mn_id.append(np.mean(mns_shifted[ds][ids:ids+step_id]))

    mn_id = np.array(mn_id)
    mn_id = (mn_id - p1_wav[2])/np.max(mn_id - p1_wav[2])
    obs_its_wrt_angles.append(mn_id)    

    p1_ret = find_ret.find_retardance(mn_id,angles)
    wav_tmp = id2wav(p1_wav[0],p1_wav[1],ids+step_id/2)
    lams.append(wav_tmp)
    modified_angles.append(angles*p1_ret[2] + p1_ret[1]*180/np.pi)
    ret[wav_tmp] = p1_ret[0]

lams = np.array(lams)

# fit for one lambda (not working)
#errfunc_modified_thick = lambda p,theta,lam, obs_its: modified_output_its(theta,lam,p[0]) - obs_its

#p0 = [1.2] #mm 

#p1_thick,success = optimize.leastsq(errfunc_modified_thick, p0[:], args=(modified_angles[174]*np.pi/180,lams[174],obs_its_wrt_angles[174]))


# surface fit (2d)
mesh_lams,mesh_angles = np.meshgrid(lams,angles)
img_obs_its = np.zeros([len(angles),len(lams)])


for i in range(len(lams)):
    for j in range(len(modified_angles[i])):
        mesh_angles[j,i] = modified_angles[i][j]
        img_obs_its[j,i] = obs_its_wrt_angles[i][j]

A = np.zeros([len(angles),len(lams),3])
A[:,:,0] = mesh_lams
A[:,:,1] = mesh_angles/180*np.pi
A[:,:,2] = img_obs_its

A = A.reshape(len(lams)*len(angles),3)



def modified_output_its(xdata,thick): #thick in mm, lam in nm
    lam = xdata[:,0]
    theta = xdata[:,1]
    ne = ne_c(lam)
    no = no_c(lam)
    phi_e = modified_retardance(ne,thick,lam)
    phi_o = modified_retardance(no,thick,lam)
    phi = phi_e - phi_o

    te = transmittance(ne,thick,lam)
    to = transmittance(no,thick,lam)
    its_tmp = te*np.cos(theta)**4 + to*np.sin(theta)**4 + 2*(te*to)**(1/2)*np.sin(theta)**2*np.cos(theta)**2*np.cos(phi)
    its =  its_tmp/np.max(its_tmp)
    return its

optimize.curve_fit(modified_output_its, A[:,:2], A[:,2], [1.1])

# wavelength resolution

def gauss_kern(sigma,size=None):
    if size == None:
        size = int(sigma)
    #size = int(size)
    x = np.arange(-size,size+1)
    g = np.exp(-(x**2/sigma))
    g = g/np.sum(g)
    return g

sim_img = modified_output_its(A[:,:2],1.145).reshape(len(angles),len(lams))
gs = gauss_kern(30,100)

sim_img_smooth = np.zeros([len(angles),len(lams)])

for i in range(len(angles)):
    sim_img_smooth[i,:] = np.convolve(sim_img[i,:],gs)[100:-100]



