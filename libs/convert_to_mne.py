import numpy as np
import math as m
from libs import utils,graph_processing as gp
from scipy.spatial.transform import Rotation as R
import mne
from libs import visualisations as viz
import matplotlib.pyplot as plt
norm = np.linalg.norm
import pandas as pd
import warnings

from scipy import optimize
def wrap_vector_dot(pos):
    def dot(params):
        a21,a22,a23,a31,a32,a33 = params
        #a1 = np.array([a11,a12,a13])
        a2 = np.array([a21,a22,a23])
        a3 = np.array([a31,a32,a33])
        np.cross(a2,a3)
        return -vector_dot(a3,pos)
    return dot

def wrap_ortho_const(ortho_axis):
    def ortho_constraint(params):
        a21,a22,a23,a31,a32,a33 = params
        a2 = np.array([a21,a22,a23])
        a3 = np.array([a31,a32,a33])
        a1 = np.cross(a2,a3)
        return np.abs(vector_dot(a2,ortho_axis))
    return ortho_constraint

def ortho_constraint12(params):
    a21,a22,a23,a31,a32,a33 = params
    a2 = np.array([a21,a22,a23])
    a3 = np.array([a31,a32,a33])
    a1 = np.cross(a2,a3)
    return a1@a2

def ortho_constraint13(params):
    a21,a22,a23,a31,a32,a33 = params
    a2 = np.array([a21,a22,a23])
    a3 = np.array([a31,a32,a33])
    a1 = np.cross(a2,a3)
    return a1@a3

def ortho_constraint23(params):
    a21,a22,a23,a31,a32,a33 = params
    a2 = np.array([a21,a22,a23])
    a3 = np.array([a31,a32,a33])
    a1 = np.cross(a2,a3)
    return a2@a3

def wrap_sum_constraint(coilori):
    def constraint(params):
        a21,a22,a23,a31,a32,a33 = params
        #a1 = np.array([a11,a12,a13])
        a2 = np.array([a21,a22,a23])
        a3 = np.array([a31,a32,a33])
        a1 = np.cross(a2,a3)
        return a1+a2+a3-coilori
    return constraint

def unit_vec_const_a1(params):
    a21,a22,a23,a31,a32,a33 = params
    a2 = np.array([a21,a22,a23])
    a3 = np.array([a31,a32,a33])
    a1 = np.cross(a2,a3)
    return np.linalg.norm(a1)-1

def unit_vec_const_a2(params):
    a21,a22,a23,a31,a32,a33 = params
    a2 = np.array([a21,a22,a23])
    a3 = np.array([a31,a32,a33])
    a1 = np.cross(a2,a3)
    return np.linalg.norm(a2)-1

def unit_vec_const_a3(params):
    a21,a22,a23,a31,a32,a33 = params
    a2 = np.array([a21,a22,a23])
    a3 = np.array([a31,a32,a33])
    a1 = np.cross(a2,a3)
    return np.linalg.norm(a3)-1


def make_optimal_orthogonal(coilori,pos,ortho_axis='x'):
    params0 = {'x':[np.array([[0,1,0],[0,0,1]]),np.array([1,0,0])],
               'y':[np.array([[1,0,0],[0,0,1]]),np.array([0,1,0])],
               'z':[np.array([[0,1,0],[1,0,0]]),np.array([0,0,1])]
               }

    res = optimize.minimize(wrap_vector_dot(pos),params0[ortho_axis][0],
                        constraints=[
        {'type':'eq','fun':wrap_ortho_const(params0[ortho_axis][1])},
        {'type':'eq','fun':ortho_constraint23},
        {'type':'eq','fun':unit_vec_const_a2},
        {'type':'eq','fun':unit_vec_const_a3}])
        
    print('success: ',res['success'])
    KIThat = res['x'].reshape(2,3)
    a1     = np.cross(KIThat[0],KIThat[1])
    KIThat = np.stack([a1,KIThat[0],KIThat[1]],axis=0)  
    return KIThat


def make_analytic_orthogonal(coilori,pos,ortho_axis='x'):
    a31,a32,a33 = coilori[0],coilori[1],coilori[2]
    if ortho_axis=='x':
        # with a22 = 1, a21=0
        # dot(a3,a2)==0 -> a23 = -a32/(a33)
        a23hat = - a32/a33
        a2 = np.array([0,1,a23hat])
    elif ortho_axis=='y':
        # with a21 = 1, a22=0
        # dot(a3,a2)==0 -> a23 = -a31/(a33)
        a23hat = - a31/a33
        a2 = np.array([1,0,a23hat])
    elif ortho_axis=='z':
        # with a23 = 0, a22=1
        # dot(a3,a2)==0 -> a21 = -a31/(a33)
        a21hat = - a32/a31
        a2 = np.array([a21hat,1,0])
    a2 = unit_vec(a2)
    a1 = np.cross(a2,coilori)
    return np.array([a1,a2,coilori])
    
def make_span(normal):
    u11=0
    u12=1
    u13 = - normal[1]*u12/normal[2]
    u1=np.array([u11,u12,u13])
    u21=1
    u22=0
    u23 = - normal[0]*u21/normal[2]
    u2=np.array([u21,u22,u23])
    return u1,u2


def project(x,u,w,r0):
    """
    x - vector to project
    u span 1
    w span 2 ortho to u
    r0 offset to 0,0,0 -> normal vector
    """
    return r0+(x-r0)/(u@u)*u+(x-r0)/(w@w)*w

def orthogonalbase(u1,u2):
    w = u2-u2@u1/(u1@u1)*u1
    return u1,w

def backproject(p,u,w,v):
    """
    p - point in 2d space
    u - base 1 of plane
    w - base 2 of plane
    v - orthovector defining plane
    """
    return p[0]*u+p[1]*w+v

def splitOri(v,dtheta=0):
    """
    v: orientation of MEG sensor
    """
    v = v/norm(v)*np.sqrt(3)
    pi = np.pi
    p_0deg   = np.array([np.cos( 2*pi*(0+dtheta)/360),  np.sin(2*pi*(0+dtheta)/360)])*np.sqrt(6)
    p_120deg = np.array([np.cos( 2*pi*(120+dtheta)/360),np.sin(2*pi*(120+dtheta)/360)])*np.sqrt(6)
    p_240deg = np.array([np.cos( 2*pi*(240+dtheta)/360),np.sin(2*pi*(240+dtheta)/360)])*np.sqrt(6)
    
    u1,u2 = make_span(v)
    u1,w = orthogonalbase(u1,u2)
    
    w = w / norm(w)
    u1 = u1 / norm(u1)

    
    a1back = backproject(p_0deg,u1,w,v)
    a2back = backproject(p_120deg,u1,w,v)
    a3back = backproject(p_240deg,u1,w,v)
    return unit_vec(a1back),unit_vec(a2back),unit_vec(a3back)
def unit_vec(vec):
    assert len(vec.shape)==1
    return vec/np.linalg.norm(vec)

def splitOri_ortho_constraint(v,orthogonal_to,delta_theta,eps=.1,theta_max=120):
    e1_prev,e2_prev,e3_prev = splitOri(v,0)
    is_smaller_e1_prev = (winkel(e1_prev,orthogonal_to)-90)>0
    is_smaller_e2_prev = (winkel(e2_prev,orthogonal_to)-90)>0
    is_smaller_e3_prev = (winkel(e3_prev,orthogonal_to)-90)>0
    w1,w2,w3=[],[],[]
    thetas = np.arange(-1,theta_max,delta_theta)
    R={'e1':[],'e2':[],'e3':[]}
    theta_hats = {'e1':[],'e2':[],'e3':[]}
    for theta_hat in thetas:
        e1,e2,e3 = splitOri(v,theta_hat)
        is_smaller_e1 = (winkel(e1,orthogonal_to)-90)>0
        is_smaller_e2 = (winkel(e2,orthogonal_to)-90)>0
        is_smaller_e3 = (winkel(e3,orthogonal_to)-90)>0
        
        if abs(winkel(e1,orthogonal_to)-90)<eps:#is_smaller_e1_prev!=is_smaller_e1 or 
            R['e1']+=[np.array([e2,e1,e3])]#fixed
            theta_hats['e1']+=[theta_hat]
        elif  abs(winkel(e2,orthogonal_to)-90)<eps:#is_smaller_e2_prev!=is_smaller_e2 or 
            R['e2']+=[np.array([e3,e2,e1])]#
            theta_hats['e2']+=[theta_hat]
        elif  abs(winkel(e3,orthogonal_to)-90)<eps:#is_smaller_e3_prev!=is_smaller_e3 or 
            R['e3']+=[np.array([e1,e3,e2])]
            theta_hats['e3']+=[theta_hat]
        else:
            is_smaller_e1_prev = is_smaller_e1
            is_smaller_e2_prev = is_smaller_e2
            is_smaller_e3_prev = is_smaller_e3
            w1+=[winkel(e1,orthogonal_to)]
            w2+=[winkel(e2,orthogonal_to)]
            w3+=[winkel(e3,orthogonal_to)]  
    #print(R)
    return R,theta_hats


def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = m.sqrt(XsqPlusYsq + z**2)               # r
    elev = m.atan2(z,m.sqrt(XsqPlusYsq))     # theta
    az = m.atan2(y,x)                           # phi
    return r, elev, az

def cart2sphA(pts):
    return np.array([cart2sph(x,y,z) for x,y,z in pts])

def appendSpherical(xyz):
    np.hstack((xyz, cart2sphA(xyz)))
    
def vector_dot(p2,p3):
    return (p2@p3)/norm(p2,axis=-1)/norm(p3,axis=-1)
    
def winkel(p2,p3):
    p2 = np.array(p2)
    p3 = np.array(p3)
    return np.degrees(np.arccos(vector_dot(p2,p3)))

def get_orthogonality(raw,verbose=False):
    """
    Get axis to which the local channel coordinate system is orthogonal to.
    A coordinate system is orthogonal to another if at least one axis is system A is
    orthogonal to one axis is system B.
    
    raw mne python object
    """
    if type(raw)==mne.io.array.array.RawArray or type(raw)==mne.io.kit.kit.RawKIT:
        pos = np.array([raw.info['chs'][i]['loc'][:3] for i in range(160)])
        ex_sens,ey_sens,ez_sens=np.array([raw.info['chs'][i]['loc'][3:].reshape(3,3) for i in range(160)]).transpose([1,0,2])
    elif type(raw)==list:
        assert len(raw)==2
        pos =raw[0]
        ex_sens,ey_sens,ez_sens =raw[1]
    else:
        raise ValueError
    ex_device = np.array([1,0,0])
    ey_device = np.array([0,1,0])
    ez_device = np.array([0,0,1])
    ortho_to = {'x':[],'y':[],'z':[]}
    
    for i in range(len(pos)):
        if verbose:
            np.set_printoptions(precision=4)
            print('Channel ID i = ',i)
            print(ey_sens[i])
            print(ex_sens[i]@ey_sens[i],ex_sens[i]@ez_sens[i],ez_sens[i]@ey_sens[i])
            print('angle e1[i] to ex,ey,ez: ',winkel(ex_device,ex_sens[i]),winkel(ey_device,ex_sens[i]),winkel(ez_device,ex_sens[i]))
            print('angle e2[i] to ex,ey,ez: ',winkel(ex_device,ey_sens[i]),winkel(ey_device,ey_sens[i]),winkel(ez_device,ey_sens[i]))
            print('angle e3[i] to ex,ey,ez: ',winkel(ex_device,ez_sens[i]),winkel(ey_device,ez_sens[i]),winkel(ez_device,ez_sens[i]))
            print('\n')
            np.set_printoptions(precision=None)
        if winkel(ey_device,ey_sens[i])==90:
            ortho_to['y']+=[(i,pos[i],ex_sens[i]+ey_sens[i]+ez_sens[i])]
            continue
        if winkel(ex_device,ey_sens[i])==90:
            ortho_to['x']+=[(i,pos[i],ex_sens[i]+ey_sens[i]+ez_sens[i])]
            continue
        if winkel(ez_device,ey_sens[i])==90:
            ortho_to['z']+=[(i,pos[i],ex_sens[i]+ey_sens[i]+ez_sens[i])]
            continue
        
            
    assert len(ortho_to['x'])+len(ortho_to['y'])+len(ortho_to['z'])==160
    return ortho_to


def make_orthogonal(chanori,i,use_axis=None,dtheta=0.5):
    eps=0.1
    if use_axis=='y' or use_axis=='all':
        R_hat,theta_hat = splitOri_ortho_constraint(chanori[i],np.array([0,1,0]),dtheta,eps=eps)
        while True:
            R_hat,theta_hat = splitOri_ortho_constraint(chanori[i],np.array([0,1,0]),dtheta,eps=eps)
            if (len(R_hat['e1'])>1 or len(R_hat['e2'])>1 or len(R_hat['e3'])>1):
                eps = eps/2
            else:
                R_hat,theta_hat = splitOri_ortho_constraint(chanori[i],np.array([0,1,0]),dtheta,eps=eps*2)
                break
        return R_hat,theta_hat
    if use_axis=='x' or use_axis=='all':
        R_hat,theta_hat = splitOri_ortho_constraint(chanori[i],np.array([1,0,0]),dtheta,eps=eps)
        while True:
            R_hat,theta_hat = splitOri_ortho_constraint(chanori[i],np.array([1,0,0]),dtheta,eps=eps)
            if (len(R_hat['e1'])>1 or len(R_hat['e2'])>1 or len(R_hat['e3'])>1):
                eps = eps/2
            else:
                R_hat,theta_hat = splitOri_ortho_constraint(chanori[i],np.array([1,0,0]),dtheta,eps=eps*2)
                break
        return R_hat,theta_hat
    if use_axis=='z' or use_axis=='all':
        R_hat,theta_hat = splitOri_ortho_constraint(chanori[i],np.array([0,0,1]),dtheta,eps=eps)
        while True:
            R_hat,theta_hat = splitOri_ortho_constraint(chanori[i],np.array([0,0,1]),dtheta,eps=eps)
            if (len(R_hat['e1'])>1 or len(R_hat['e2'])>1 or len(R_hat['e3'])>1):
                eps = eps/2
            else:
                R_hat,theta_hat = splitOri_ortho_constraint(chanori[i],np.array([0,0,1]),dtheta,eps=eps*2)
                break
        return R_hat,theta_hat
    print('fail')
    return splitOri(chanori[i],dtheta=0),-1



def similarize_point_clouds(chanpos,raw_reference,site):
    chanpos = chanpos*1000
    chpos_reference = np.array([raw_reference.info['chs'][i]['loc'][:3]*1000 for i in range(160)])
    from scipy.spatial.transform import Rotation as R
    if '10020' in raw_reference.info['description']:
        if site=='A':
            shift_x = 0
            shift_y = -20
            shift_z = +20
        else:
            shift_x = -4
            shift_y = -8
            shift_z = 22
        rz = R.from_rotvec(np.radians(4) * np.array([0,0,1]))
        chanpos = chanpos@rz.as_matrix()
        ry = R.from_rotvec(np.radians(-1) * np.array([0,1,0]))
        chanpos = chanpos@ry.as_matrix()
        rx = R.from_rotvec(np.radians(2) * np.array([1,0,0]))
        chanpos = chanpos@rx.as_matrix()
    elif '10021' in raw_reference.info['description']:
        if site=='A':
            shift_x = 0
            shift_y = 0
            shift_z = 0
        else:
            shift_x = 0
            shift_y = -80
            shift_z = 20
            rz = R.from_rotvec(np.radians(5) * np.array([0,0,1]))
            chanpos = chanpos@rz.as_matrix()          
    elif "V3R000 PQA160C" in raw_reference.info['description']:
        if site=='A':
            shift_x = 0
            shift_y = 0
            shift_z = 20 
            rz = R.from_rotvec(np.radians(-5) * np.array([0,0,1]))
            chanpos = chanpos@rz.as_matrix()
        else:
            shift_x = 0
            shift_y = 0
            shift_z = +20 
            rz = R.from_rotvec(np.radians(5) * np.array([0,0,1]))
            chanpos = chanpos@rz.as_matrix()
    elif "V2R004 PQA160C" in raw_reference.info['description']:
        if site=='A':
            shift_x = 0
            shift_y = 0
            shift_z = 20 
        else:
            shift_x = 0
            shift_y = 0
            shift_z = 20 
    else:
        shift_x = 0
        shift_y = 0
        shift_z = 0 
    chanpos[:,0]=chanpos[:,0]+shift_x
    chanpos[:,1]=chanpos[:,1]+shift_y
    chanpos[:,2]=chanpos[:,2]+shift_z
    return chanpos,chpos_reference 

def match_to_reference(chanpos,chpos_ref):
    return gp.matchKITsystems(chanpos,chpos_ref)

def get_site(fs):
    if fs==1000:
        return 'A'
    else:
        return 'B'
def select_channel_system(systems):
    if len(systems)==0:
        return []
    else:
        return systems[int(len(systems)//2)]

def ortho_from_index(ortho_to,i):
    try:
        return pd.DataFrame(np.array(ortho_to['x'])[:,1],np.array(ortho_to['x'])[:,0],columns=['x']).loc[i]
    except:
        try:
            return pd.DataFrame(np.array(ortho_to['y'])[:,1],np.array(ortho_to['y'])[:,0],columns=['y']).loc[i]
        except:
            return pd.DataFrame(np.array(ortho_to['z'])[:,1],np.array(ortho_to['z'])[:,0],columns=['z']).loc[i]    
   
def run(meg,infospm,rawcon,verbose=False):
    fs=int(utils.load_key_chain(infospm['D'],['Fsample']))
    site = get_site(fs)
    
    markers = utils.load_key_chain(infospm['D'],['fiducials','fid','pnt'])/1000
    chanpos = utils.load_key_chain(infospm['D'],['sensors','meg','chanpos'])/1000
    chanori = utils.load_key_chain(infospm['D'],['sensors','meg','chanori'])
    markers[:,[0, 1]] = markers[:,[1, 0]]
    chanori[:,[0, 1]] = chanori[:,[1, 0]]
    chanpos[:,[0, 1]] = chanpos[:,[1, 0]] 


    def sideway(chanpos,rawcon,site):
        chanpos,chpos_reference = similarize_point_clouds(chanpos,rawcon,site)

        matching,greedy_match = match_to_reference(chanpos,chpos_reference)
        if verbose:
            viz.plot_with_reference_sytem(chanpos,chpos_reference)
            viz.compare_matching(chanpos,chpos_reference,matching,greedy_match)
        max_dist = np.max(np.linalg.norm(chpos_reference[matching]-chanpos,axis=-1))
        # algorithm needed that selects all outliers
        if site=='A':
            outlier = [np.argmax(np.linalg.norm(chpos_reference[matching]-chanpos,axis=-1))]# idx in chanpos
        else:
            outlier = []
        return matching,greedy_match,outlier

    matching,greedy_match,outlier = sideway(chanpos,rawcon,site)

    ch_pos = {'MEG {:03}'.format(ch):chanpos[ch] for ch in range(160)}
    digmontage = mne.channels.make_dig_montage(ch_pos=ch_pos,nasion=markers[0],lpa=markers[1],rpa=markers[2])
    info = mne.create_info(ch_names=['MEG {:03}'.format(ch) for ch in range(160)], sfreq=fs, ch_types='grad')
    info.set_montage(digmontage)
    raw = mne.io.RawArray(meg, info)
    
    fsResample = 1000
    raw.filter(0.,400)
    warnings.warn('Resampling signal to {} Hz. Take care when using epoched data'.format(fsResample))
    fs = fsResample
    raw.resample(fsResample)

    
    #https://www.fieldtriptoolbox.org/faq/how_are_the_different_head_and_mri_coordinate_systems_defined/ 
    # fiducial points: - in the helmet’s own coordinate system-> dev to head is identity transform
    raw.info['dev_head_t'] = rawcon.info['dev_head_t']
    raw.info['dev_head_t']['trans'] = np.eye(4)
    #raw.info['dev_head_t']['trans'][2,3]=-0.01

    mne_cal      = rawcon.info['chs'][0]['cal']
    mne_range    = rawcon.info['chs'][0]['range']
    mne_unit     = rawcon.info['chs'][0]['unit']
    mne_unit_mul = rawcon.info['chs'][0]['unit_mul']
    # ch_name
    mne_ch_coord_frame   = rawcon.info['chs'][0]['coord_frame']
    mne_ch_coil_type_KIT = rawcon.info['chs'][0]['coil_type']
    mne_ch_kind          = rawcon.info['chs'][0]['kind']
    # ch_loc    
    last_axis_outwards = []
    for i in range(160):

        raw.info['chs'][i]['cal']         = mne_cal
        raw.info['chs'][i]['logno']       = i+1
        raw.info['chs'][i]['scanno']      = i+1
        raw.info['chs'][i]['range']       = mne_range
        raw.info['chs'][i]['unit']        = mne_unit
        raw.info['chs'][i]['unit_mul']    = mne.utils._bunch.NamedInt('FIFF_UNITM_F',-15)#mne_unit_mul
        raw.info['chs'][i]['coil_type']   = mne_ch_coil_type_KIT
        raw.info['chs'][i]['coord_frame'] = mne_ch_coord_frame
        raw.info['chs'][i]['kind']        = mne_ch_kind
        if verbose:
            print('coil direction: ', chanori[i])
            print('coil position: ', chanpos[i])
        
        ortho_to = get_orthogonality(rawcon,verbose=False)
        orth = ortho_from_index(ortho_to,greedy_match[i]).index[0]
        A=make_analytic_orthogonal(unit_vec(chanori[i]),[],ortho_axis=orth)
        loc = np.concatenate([chanpos[i:i+1],A]).flatten()
        raw.info['chs'][i]['loc']         = loc
    print('site', site)
    if site=='A':
        sphere = mne.make_sphere_model(r0=(0,0,0.),info=raw.info,head_radius=np.linalg.norm(markers[1]-markers[2])/2)
    else:
        sphere = mne.make_sphere_model(r0=(0,0,-0.01),info=raw.info,head_radius=np.linalg.norm(markers[1]-markers[2])/2)
    if verbose:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        vis.plotIgel(raw,ax,ortho_to)
        ax.view_init(24.1558436, 49.090)
    
    return raw,sphere   
    

"""
if i==5:
        print('KIT channel: ')
        print(ch_coord_from_KIT.reshape(3,3))
        print('pos: ', chanpos[i:i+1])
        print('ori: ',np.sum(ch_coord_from_KIT.reshape(3,3),axis=0))
        viz.plot_sensor_coord_system(ch_coord_from_KIT.reshape(3,3),'KIT')
    
        ch_ortho_x=make_orthogonal(chanori,i,'x',dtheta=0.2)[0]
        ch_ortho_y=make_orthogonal(chanori,i,'y',dtheta=0.2)[0]
        ch_ortho_z=make_orthogonal(chanori,i,'z',dtheta=0.05)[0]
        
        
        print("ch_ortho_x['e1']:\n", select_channel_system(ch_ortho_x['e1']))
        print("ch_ortho_x['e2']:\n", select_channel_system(ch_ortho_x['e2']))
        print("ch_ortho_x['e3']:\n", select_channel_system(ch_ortho_x['e3']))
        
        print("ch_ortho_y['e1']:\n", select_channel_system(ch_ortho_y['e1']))
        print("ch_ortho_y['e2']:\n", select_channel_system(ch_ortho_y['e2']))
        print("ch_ortho_y['e3']:\n", select_channel_system(ch_ortho_y['e3']))
        
        print("ch_ortho_z['e1']:\n", select_channel_system(ch_ortho_z['e1']))
        print("ch_ortho_z['e2']:\n", select_channel_system(ch_ortho_z['e2']))
        print("ch_ortho_z['e3']:\n", select_channel_system(ch_ortho_z['e3']))
        #viz.plot_sensor_coord_system(select_channel_system(ch_ortho_x['e1']),'xe1')#2
        #viz.plot_sensor_coord_system(select_channel_system(ch_ortho_x['e2']),'xe2')#3
        
        #viz.plot_sensor_coord_system(select_channel_system(ch_ortho_y['e2']),'ye2')
        #viz.plot_sensor_coord_system(select_channel_system(ch_ortho_y['e3']),'ye3')
        
        #viz.plot_sensor_coord_system(select_channel_system(ch_ortho_z['e1']),'ze1')
        #viz.plot_sensor_coord_system(select_channel_system(ch_ortho_z['e3']),'ze3')
        
else:
    print('outlier', i)
    # Das Erzeugendensystem chanori sei e1,e2,e3 dann ist chanori=(e1+e2+e3)/np.sqrt(3)
    # Die Menge der Ezeugendensystem ist jede moegliche Drehung um chanori
    # In den KIT Systemen ist eine Achse des Erzeugendensystems orhogonal zu ex oder ey oder ez
"""