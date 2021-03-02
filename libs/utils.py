#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Utility function for the BioMag Dementia Challenge.

(c) 2021 Marc Steffen Seibel, Technische Universitaet Ilmenau
'''


from difflib import get_close_matches
from distutils.version import LooseVersion
import operator
import os
import os.path as op
import sys
from pathlib import Path
import numpy as np
import mne
from _ctypes import PyObj_FromPtr
import json
import re
import pip
import pandas as pd
import warnings
from scipy import io
import pathlib
import functools

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def split_subject_dict(subjects):
    num_subjects = len(subjects['control'])+len(subjects['dementia'])+len(subjects['mci'])
    small_subjects = []
    counter = 0
    for k in subjects.keys():
        for subid in subjects[k]:
            cd = {'control':[],'dementia':[],'mci':[]}
            cd[k] = [subid]
            small_subjects+=[cd]
            counter+=1
    return small_subjects

def load_bad_samples(condition,subject,path=r"E:\2021_Biomag_Dementia_MNE\inT_naive_resample\bad_samples"):
    bads = np.load(os.path.join(path,'bad_samples{}{}.npy'.format(condition,subject)))
    bads = np.unpackbits(bads,axis=None).reshape(160,300000).astype(np.bool)
    return bads
    
def ema(data,alpha):
    """
    in place ema
    """
    num_samples = data.shape[1]
    #ma = np.mean(data,axis=1,keepdims=True)
    for t in np.arange(1,num_samples):
        data[:,t] = alpha*data[:,t]+(1-alpha)*data[:,t-1]
    return data
    
def emstd(data,ema,alpha):
    """
    inplace std
    """
    num_samples = data.shape[1]
    #ma = np.mean(data,axis=1,keepdims=True)
    data[:,0]=np.var(data,axis=1)
    for t in np.arange(1,num_samples):
        data[:,t] = alpha*(data[:,t]-ema[:,t])**2+(1-alpha)*data[:,t-1]
    return np.sqrt(data)
    

def ema_substraction(data,alpha):
    """
    data: channels,samples
    small alpha means large memory for the mean
    """
    moving_avg = data.copy()    
    num_samples = data.shape[1]
    #ma = np.mean(data,axis=1,keepdims=True)
    ema(moving_avg,alpha=alpha)
    return data-moving_avg

def em_avg_std(data,alpha):
    """
    data: channels,samples
    small alpha means large memory for the mean
    """
    num_samples = data.shape[1]
    #ma = np.mean(data,axis=1,keepdims=True)
    moving_avg = ema(data.copy(),alpha=alpha)
    em_std = emstd(data.copy(),moving_avg,alpha)
    return (data-moving_avg)/em_std




def moving_average(data,N,padding=0,alpha=1):
    """
    data.shape: (channels,time_steps)
    """
    num_samples = data.shape[1]
    cleaned = data.copy()
    EMA = np.mean(data[:,0:N])
    for k in np.arange(num_samples-N):
        EMA = alpha*np.mean(data[:,k:k+N])+(1-alpha)*EMA
        cleaned[:,k:k+N]=data[:,k:k+N]-EMA
    return cleaned
    
def _moving_average_ch(data,N,padding=0,alpha=1):
    """
    data.shape: (time_steps)
    """
    num_samples = data.shape[0]
    cleaned = data.copy()
    EMA = np.mean(data[0:N])
    for k in np.arange(num_samples-N):
        EMA = alpha*np.mean(data[k:k+N])+(1-alpha)*EMA
        cleaned[k:k+N]=data[k:k+N]-EMA
    return cleaned
def get_class_labels(tasks):
    mapping = {'dementia':'dem','control':'control',
        'mci':'mci'}
    labels=[]
    for t in tasks:
        labels+=[mapping[t]]
    return labels
    
def parse_filename(filename):
    conditions = ['control','dementia','mci']
    for c in conditions:
        if c in filename:
            _,remainder = filename.split(c)
            condition = c
            subject_id = int(''.join(filter(lambda x: x.isdigit(), remainder)))
            break
    return condition,subject_id
    
def serialize_np(items):
    return items.tolist()

def suject_dicts_are_unique(subject_dicts):
    control_subjects  = np.concatenate([fold_dict['control'] for fold_dict in subject_dicts]).astype(int)
    dementia_subjects = np.concatenate([fold_dict['dementia'] for fold_dict in subject_dicts]).astype(int)
    mci_subjects      = np.concatenate([fold_dict['mci'] for fold_dict in subject_dicts]).astype(int)
    any_double = 0
    any_double += np.sum(np.bincount(control_subjects)>1)
    any_double += np.sum(np.bincount(dementia_subjects)>1)
    any_double += np.sum(np.bincount(mci_subjects)>1)
    if any_double!=0:
        print('control:\n',control_subjects)
        print('dementia:\n',dementia_subjects)
        print('mci:\n',mci_subjects)
        return False
    else:
        return True

def check_dicts_contain_subjects(subject_dicts,subjects=None):
    """
    If subjects is None that all subjects of the biomag dataset must be contained in subject_dicts
    """    
    return

def split_subjects(subjects,method,**kwargs):
    return method(subjects,**kwargs)

class Subject_Splitter():
    def __init__(self,subjects,method,**kwargs):
        gen_split_A,gen_split_B,subjectsA,subjectsB, labelsA, labelsB
        self.generator = zip(gen_split_A,gen_split_B)
        return
    def __getitem__(self,k):
        return 

def subjects_by_site(subjects):
    assert len(np.setdiff1d(['mci','dementia','control'],list(subjects.keys())))==0,(subjects.keys())
    subjects_by_site = {'A':{'mci':[],'dementia':[],'control':[]},
                        'B':{'mci':[],'dementia':[],'control':[]}}
    for condition in list(subjects.keys()):
        by_site = sort_by_site(subjects,condition,path= r'./dataframes/maxwell_and_temporal')
        subjects_by_site['A'][condition].extend(by_site['A'])
        subjects_by_site['B'][condition].extend(by_site['B'])
    return subjects_by_site
            
def split_wrt_site(subjects,test_ratio):
    from sklearn.model_selection import train_test_split
    if not 'A' in subjects.keys() or not 'B' in subjects.keys():
        subjects = subjects_by_site(subjects)
    labelsA,labelsB, subjectsA, subjectsB = _split_wrt_site_base(subjects)
    subjectsA_train,subjectsA_test,labelsA_train,labelsA_test = train_test_split(
        subjectsA,labelsA,test_size=test_ratio)
    subjectsB_train,subjectsB_test,labelsB_train,labelsB_test = train_test_split(
        subjectsB,labelsB,test_size=test_ratio)
        
    print(np.where(labelsB_train=='control'))
    traincontrol  = np.concatenate([subjectsA_train[labelsA_train=='control']  , subjectsB_train[labelsB_train=='control']])
    traindementia = np.concatenate([subjectsA_train[labelsA_train=='dementia'] , subjectsB_train[labelsB_train=='dementia']])
    trainmci      = np.concatenate([subjectsA_train[labelsA_train=='mci']      , subjectsB_train[labelsB_train=='mci']])
    testcontrol  = np.concatenate([subjectsA_test[labelsA_test=='control']  , subjectsB_test[labelsB_test=='control']])
    testdementia = np.concatenate([subjectsA_test[labelsA_test=='dementia'] , subjectsB_test[labelsB_test=='dementia']])
    testmci      = np.concatenate([subjectsA_test[labelsA_test=='mci']      , subjectsB_test[labelsB_test=='mci']])
        
    subjects_train =  {'control':traincontrol,'mci':trainmci,'dementia':traindementia}
    subjects_test  =  {'control':testcontrol,'mci':testmci,'dementia':testdementia}
    return subjects_train, subjects_test
    
def _split_wrt_site_base(subjects):
    subjectsA = subjects['A']
    subjectsB = subjects['B']
    
    subject_controlA = subjectsA['control']
    subject_dementA = subjectsA['dementia']
    subject_mciA = subjectsA['mci']
    subjectsA = np.concatenate([subject_controlA,subject_dementA,subject_mciA])
    labelsA = len(subject_controlA)*['control']+\
              len(subject_dementA)*['dementia']+\
              len(subject_mciA)*['mci']
    
    subject_controlB = subjectsB['control']
    subject_dementB = subjectsB['dementia']
    subject_mciB = subjectsB['mci']
    subjectsB = np.concatenate([subject_controlB,subject_dementB,subject_mciB])
    labelsB = len(subject_controlB)*['control']+\
              len(subject_dementB)*['dementia']+\
              len(subject_mciB)*['mci']
    return np.array(labelsA),np.array(labelsB), subjectsA, subjectsB
    
def _subjects_dict_wrt_site(subjects):
    subjectsA = subjects['A']
    subjectsB = subjects['B']
    return subjectsA, subjectsB
    
def sort_by_site(subjects,condition,path= r'./dataframes/maxwell_and_temporal'):
    """
    sorts subject with a condition by site
    """
    by_site = {'A':[],'B':[]}
    for k in subjects[condition]:
        site = get_site_from_condition_number(condition,k,path)
        by_site[site]+=[k]
    by_site['A']=np.array(by_site['A'])
    by_site['B']=np.array(by_site['B'])
    return by_site
    

def cv_split_wrt_site(subjects,n_splits):
    """
    subjects has keys: ['A','B']
    and values being dicts with keys 'control', 'dementia', 'mci'
    returns all splits
    """
    from sklearn.model_selection import StratifiedKFold
    if not 'A' in subjects.keys() or not 'B' in subjects.keys():
        subjects = subjects_by_site(subjects)
    labelsA,labelsB, subjectsA, subjectsB = _split_wrt_site_base(subjects)
    
    kfoldA      = StratifiedKFold(n_splits=n_splits)
    kfoldB      = StratifiedKFold(n_splits=n_splits)
    gen_split_A = kfoldA.split(np.arange(len(labelsA)),labelsA)
    gen_split_B = kfoldB.split(np.arange(len(labelsB)),labelsB)
    splits = []
    for k in range(n_splits):
        
        trainidxA,testidxA = next(gen_split_A)
        trainidxB,testidxB = next(gen_split_B)
        labelsAtrain = labelsA[trainidxA]
        labelsBtrain = labelsB[trainidxB]
        labelsAtest = labelsA[testidxA]
        labelsBtest = labelsB[testidxB]
               
        traincontrol   = np.concatenate([subjectsA[trainidxA][labelsAtrain=='control'],
                                         subjectsB[trainidxB][labelsBtrain=='control']])
        traindementia  = np.concatenate([subjectsA[trainidxA][labelsAtrain=='dementia'],
                                         subjectsB[trainidxB][labelsBtrain=='dementia']])
        trainmci       = np.concatenate([subjectsA[trainidxA][labelsAtrain=='mci'],
                                         subjectsB[trainidxB][labelsBtrain=='mci']])
                                         
        testcontrol   = np.concatenate([subjectsA[testidxA][labelsAtest=='control'],
                                        subjectsB[testidxB][labelsBtest=='control']])
        testdementia  = np.concatenate([subjectsA[testidxA][labelsAtest=='dementia'],
                                        subjectsB[testidxB][labelsBtest=='dementia']])
        testmci       = np.concatenate([subjectsA[testidxA][labelsAtest=='mci'],
                                        subjectsB[testidxB][labelsBtest=='mci']])                                       
        subjects_train =  {'control':traincontrol,'mci':trainmci,'dementia':traindementia}
        subjects_test  =  {'control':testcontrol,'mci':testmci,'dementia':testdementia}   
        splits+=[(subjects_train,subjects_test)]
    return splits
    
def split_ratio(subjects,test_ratio):
    """
    test_ratio percentage of data from each class which is left out from training and put aside for testing.
    """
    subjects_dementia = subjects['dementia']
    subjects_control = subjects['control']
    subjects_mci = subjects['mci']
    def choice(idx):
        if len(idx)==0:
            return np.array([])
        num_test = int(len(idx)*test_ratio)
        if num_test==0:
            warnings.warn('# of cases for any class is 0. Test data will be empty.')
            return []
        else:    
            return np.random.choice(idx,size=num_test,replace=False)
    testdementia = choice(subjects_dementia)
    testcontrol = choice(subjects_control)
    testmci = choice(subjects_mci)
    traindementia = np.array(list(set(list(subjects_dementia))-set(list(testdementia))))
    traincontrol = np.array(list(set(list(subjects_control))-set(list(testcontrol))))
    trainmci = np.array(list(set(list(subjects_mci))-set(list(testmci))))
    return {'control':traincontrol,'mci':trainmci,'dementia':traindementia},\
        {'control':testcontrol,'mci':testmci,'dementia':testdementia}

def condition_to_digit(condition):
    if condition=='mci':
        return 1
    elif condition=='control':
        return 0
    elif condition=='dementia':
        return 2
    else:
        raise ValueError("condition must be mci, control or dementia")
        
def digit_to_condition(digit):
    if digit==1:
        return 'mci'
    elif digit==0:
        return 'control'
    elif digit==2:
        return 'dementia'
    else:
        raise ValueError("condition must be mci, control or dementia")


def unique_bads_in_fif(raw):
    raw.info['bads'] = list(np.unique(np.array(raw.info['bads'])))
    
def get_subjects_wrt_site(subjects,cA,cB,condition,dataframes='dataframes/maxwell_and_temporal'):
    num_subjects = {'control':100,'mci':15,'dementia':29}[condition]
    for k in np.arange(1,num_subjects+1):
        site = get_site_from_condition_number(condition,k,dataframes)
        if site=='A' and cA>0:
            cA-=1
            subjects[condition]+=[k]
        elif site=='B' and cB>0:
            cB-=1
            subjects[condition]+=[k]
        elif cA==0 and cB==0:
            break

def get_raw_mne_info_condition_number(condition,number,path=r'E:\2021_Biomag_Dementia_MNE\inT\interpolated100Hz\raw'):
    print('site: ',get_site_from_condition_number(condition,number))
    raw = mne.io.read_raw_fif(os.path.join(path,'100Hz{}{:03d}raw.fif'.format(condition,number)))
    return raw.info 

def correct_rotation_info(infoB):
    from scipy.spatial.transform import Rotation as R
    locs = np.array([infoB['chs'][ch]['loc'][:3] for ch in range(160)])
    rz = R.from_rotvec(np.radians(6) * np.array([0,0,1]))
    locs = locs@rz.as_matrix()
    for ch in range(160):
        infoB['chs'][ch]['loc'][:3]=locs[ch]
    return infoB
    
def correct_rotation(rawB):
    rawB.info = correct_rotation_info(rawB.info)
    return rawB

    
def get_site_from_condition_number(condition,subject_number,direc= r'dataframes\maxwell_and_temporal'):
    warnings.warn("Method is deprecated us 'get_site'.")
    return get_site('foo',**{'condition':condition,'number':subject_number})
    assert condition in ['mci','dementia','control']
    maxwelldf = os.listdir(direc)
    site = [f for f in maxwelldf if condition in f and '{:03d}'.format(subject_number) in f]
    assert len(site)==1,(site)
    site = site[0]
    site = site.split(condition)[1].split('site')[1][0]
    assert site in ['A','B']
    return site

def get_site_from_json(condition,number,filepath='.'):
    path = pathlib.Path(filepath)
    if condition=='test':
        with open(path / 'sites_test.json','r') as f:
            site_dict = json.load(f)
    elif condition=='control':
        with open(path / 'sites_control.json','r') as f:
            site_dict = json.load(f)
    elif condition=='dementia':
        with open(path / 'sites_dementia.json','r') as f:
            site_dict = json.load(f)
    elif condition=='mci':
        with open(path / 'sites_mci.json','r') as f:
            site_dict = json.load(f)
    elif condition=='train':
        with open(path / 'sites_train.json','r') as f:
            site_dict = json.load(f)
    else:
        raise ValueError("Wrong condition provided.")
    return site_dict['{}{:03d}'.format(condition,number)]

def get_site(*args,**kwargs):
    if 'fs' in kwargs.keys():
        return get_site_from_fs(kwargs['fs'])
    elif 'condition' in kwargs.keys():
        condition=kwargs['condition']
        number = kwargs['number']
        return get_site_from_json(condition,number)
    else:
        raise ValueError(kwargs.keys())
        
def get_stable_channels(matches,order):
    """
    returns the channels that have order+1 matches
    """
    stable_chs = []
    for ch in range(160):
        unique_matches = np.unique(matches[:,1,ch])
        if len(unique_matches)==order+1:
            stable_chs+=[(ch,unique_matches)]
    
    # matches site A and site B have same shape for order 0
    stable_chs = np.array(stable_chs)
    if order==0:
        stable_chs[:,1] = np.concatenate(stable_chs[:,1]).astype(np.uint8)
    return stable_chs.astype(np.uint8)

# used for BioMag2021 channel matching
def write_matching(node_edit_path_dict,filename):
    if not 'info' in node_edit_path_dict.keys():
        raise KeyError('node_edit_path_dict must have key "info" ')   
    with open(filename,'w') as f:
        json.dump({'matching0':node_edit_path_dict},f)

def load_matching(filename,remove_info=True):
    with open(filename,'r') as f:
        loaded_d = json.load(f)
        
    node_edit_path_dict = loaded_d['matching0']
    if remove_info:
        print('info: ',node_edit_path_dict['info'])
        del node_edit_path_dict['info']
    return np.array(list(node_edit_path_dict.items())).astype('int')



def load_key_chain(sub_info,key_chain:list):
    """
    sub_info: meta information in spm12 format e.g. from BioMag2021 Competition 
    """
    assert type(key_chain)==list,"key_chain must be of type list"
    try:
        if sub_info.shape==(1,1):
            lvl_info = sub_info.flat[0][key_chain[0]]
        else:
            lvl_info = sub_info[key_chain[0]]
    except ValueError as e:
        print(e)
        print('Possible next keys are: ',sub_info.dtype.names)
        return 
    if len(key_chain)!=1:
        return load_key_chain(lvl_info,key_chain[1:])
    #print('key loaded')
    return lvl_info.copy()
    
def get2D_coords(info):
    x_pos = load_key_chain(info['D'],['channels','X_plot2D'])
    x_pos = np.concatenate(np.squeeze(x_pos)[:160])
    y_pos = load_key_chain(info['D'],['channels','Y_plot2D'])
    y_pos = np.concatenate(np.squeeze(y_pos)[:160])
    coords2D = np.transpose(np.array([x_pos,y_pos])[:,:,0])
    return coords2D
    

        
def split_channel_groups(data,meta):
    """
    With respect to the sensor site, a different number of channels is given.
    In both sites the first 160 channels contain the meg data.
    
    params:
    -------
    data: array w/ shape (160+type2channels+type3channels,time_samples)
    meta: 
    
    returns:
    -------
    meg,type2records,type3records
    """
    
    ch_labels = load_key_chain(meta['D'],['channels'])['label'][0]
    type2trailing = ch_labels[160][0][:3] 
    meg    = data['data'][:160]
    N_type2 = np.max([j+160 for j,label in enumerate(ch_labels[160:]) if type2trailing in label[0]])+1

    type2channels = slice(160,N_type2)
    type3channels = slice(N_type2,None)
    
    type2records    = data['data'][type2channels]
    type3records = data['data'][type3channels]
    return meg,type2records,type3records
    
def load_spm_info(meta_path,subject,condition=None):
    if condition is None:
        condition='test'
    infospm = io.loadmat(meta_path +'\hokuto_{}{}.mat'.format(condition,subject))
    return infospm
    
def load_spm(conditionGroup_meta,conditionGroup_data,subject,just_meg=True):
    """
    return meg, infospm
    """
    if 'control' in conditionGroup_data:
        condition='control'
    elif 'dementia' in conditionGroup_data:
        condition='dementia'
    elif 'mci' in conditionGroup_data:
        condition='mci'
    else:
        raise NameError("name 'condition' is not defined")
    infospm = load_spm_info(conditionGroup_meta ,subject,condition)
    dataspm = io.loadmat(conditionGroup_data +'\hokuto_{}{}.mat'.format(condition,subject))
    fs=int(load_key_chain(infospm['D'],['Fsample']))
    if fs==1000:
        type2name = 'eeg'
        type3name = 'others'
        site='A'
    else:
        type2name ='trig'
        type3name = 'magnetometer reference'
        site='B'    
    meg,type2records,type3records = split_channel_groups(dataspm,infospm)
    if just_meg:
        return meg, infospm
    else:
        return meg,type2records,type3records, infospm



def get_fiducials(info):
    """
    returns:
    ------
    DataFrame of fiducial points
    """
    fid_lab_iter = np.concatenate(load_key_chain(info['D'],['fiducials','fid','label']).flatten())
    pnts_iter = load_key_chain(info['D'],['fiducials','fid','pnt'])
    return pd.DataFrame({'x':pnts_iter[:,0],'y':pnts_iter[:,1],'z':pnts_iter[:,2]},index=fid_lab_iter)
    
@deprecated
class UnitData(float):
    def __new__(cls,value,*a,**k):
        return float.__new__(cls,value)
    def __init__(self, value, *args, **kwargs):
        self.unit = kwargs.pop('unit', None)

    def __str__(self):
        return 'UnitData(value=%f, unit=%s)' % (self, self.unit)
        
    def __index__(self):
        # looks weird but enables indexing with float
        if (self-int(self))==0:
            return int(self)
        else:
            raise IndexError
            
    def __repr__(self):
        return 'UnitData(value=%f,unit=%s)' % (self, self.unit)

# taken from mne
class Bunch(dict):
    """Dictionary-like object that exposes its keys as attributes.
    Has a nice property that:
    >> t = Bunch(value=5,unit='s')
    >> t
    {'value':5, 'unit':'s'}
    >> t1 = Bunch(**t)
    >> t1
    {'value':5, 'unit':'s'}
    Therefore Bunch objects are easily json serializable.
    """
    
    def __init__(self, **kwargs):  # noqa: D102
        dict.__init__(self, kwargs)
        self.__dict__ = self



###############################################################################
# A protected version that prevents overwriting

class BunchConst(Bunch):
    """Class to prevent us from re-defining constants (DRY)."""
    def __setattr__(self, attr, val):  # noqa: D105
        if attr != '__dict__' and hasattr(self, attr):
            raise AttributeError('Attribute "%s" already set' % attr)
        super().__setattr__(attr, val)
        

def time2samples(time_container,fs):
    assert hasattr(time_container,'unit')
    assert hasattr(time_container,'value')
    
    if time_container.unit=='s':
        pass
    elif time_container.unit=='ms':
        time_container.unit='s'
        time_container.value=time_container.value/1000
    elif time_container.unit==1:
        return int(time_container.value)
    else:
        raise ValueError
    return int(time_container.value*fs)
    

def get_site_from_fs(fs):
    if fs==1000:
        site = 'A'
    elif fs==2000:
        site = 'B'
    else:
        raise ValueError("Recording site is determined from sampling\
            frequency which is either 1000 (site A) or 2000 (site B).")
    return site


class TimeData(float):
    def __new__(cls,value,*a,**k):
        obj = float.__new__(cls,value)
    
    def __init__(self, specific_arg_for_time, *args, **kwargs):
        super().__init__(kwargs)


class MEGArray(np.ndarray):
    def __new__(cls, input_array, site=None,position=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        
        # add the new attribute to the created instance
        obj.site = site
        return obj
        
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.site = getattr(obj, 'site', None)  
        

# copy pasted code to make json print beautiful: https://tinyurl.com/ycsdukjh
class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        self.value = value


def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])    

class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(MyEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

       # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)
        return json_repr


def _setup_vmin_vmax(data, vmin, vmax, norm=False):
    """Handle vmin and vmax parameters for visualizing topomaps.
    For the normal use-case (when `vmin` and `vmax` are None), the parameter
    `norm` drives the computation. When norm=False, data is supposed to come
    from a mag and the output tuple (vmin, vmax) is symmetric range
    (-x, x) where x is the max(abs(data)). When norm=True (a.k.a. data is the
    L2 norm of a gradiometer pair) the output tuple corresponds to (0, x).
    Otherwise, vmin and vmax are callables that drive the operation.
    """
    should_warn = False
    if vmax is None and vmin is None:
        vmax = np.abs(data).max()
        vmin = 0. if norm else -vmax
        if vmin == 0 and np.min(data) < 0:
            should_warn = True

    else:
        if callable(vmin):
            vmin = vmin(data)
        elif vmin is None:
            vmin = 0. if norm else np.min(data)
            if vmin == 0 and np.min(data) < 0:
                should_warn = True

        if callable(vmax):
            vmax = vmax(data)
        elif vmax is None:
            vmax = np.max(data)

    if should_warn:
        warn_msg = ("_setup_vmin_vmax output a (min={vmin}, max={vmax})"
                    " range whereas the minimum of data is {data_min}")
        warn_val = {'vmin': vmin, 'vmax': vmax, 'data_min': np.min(data)}
        #warn(warn_msg.format(**warn_val), UserWarning)

    return vmin, vmax

def _check_option(parameter, value, allowed_values, extra=''):
    """Check the value of a parameter against a list of valid options.
    Raises a ValueError with a readable error message if the value was invalid.
    Parameters
    ----------
    parameter : str
        The name of the parameter to check. This is used in the error message.
    value : any type
        The value of the parameter to check.
    allowed_values : list
        The list of allowed values for the parameter.
    extra : str
        Extra string to append to the invalid value sentence, e.g.
        "when using ico mode".
    Raises
    ------
    ValueError
        When the value of the parameter was not one of the valid options.

    @mne Github
    """
    if value in allowed_values:
        return True

    # Prepare a nice error message for the user
    extra = ' ' + extra if extra else extra
    msg = ("Invalid value for the '{parameter}' parameter{extra}. "
           '{options}, but got {value!r} instead.')
    allowed_values = list(allowed_values)  # e.g., if a dict was given
    if len(allowed_values) == 1:
        options = f'The only allowed value is {repr(allowed_values[0])}'
    else:
        options = 'Allowed values are '
        options += ', '.join([f'{repr(v)}' for v in allowed_values[:-1]])
        options += f' and {repr(allowed_values[-1])}'
    raise ValueError(msg.format(parameter=parameter, options=options,
                                value=value, extra=extra))



def ewma_vectorized_2d(data, alpha, axis=None, offset=None, dtype=None, order='C', out=None):
    """
    https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
    Calculates the exponential moving average over a given axis.
    :param data: Input data, must be 1D or 2D array.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param axis: The axis to apply the moving average on.
        If axis==None, the data is flattened.
    :param offset: optional
        The offset for the moving average. Must be scalar or a
        vector with one element for each row of data. If set to None,
        defaults to the first value of each row.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Ignored if axis is not None.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    assert data.ndim <= 2

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if axis is None or data.ndim < 2:
        # use 1D version
        if isinstance(offset, np.ndarray):
            offset = offset[0]
        return ewma_vectorized(data, alpha, offset, dtype=dtype, order=order,
                               out=out)

    assert -data.ndim <= axis < data.ndim

    # create reshaped data views
    out_view = out
    if axis < 0:
        axis = data.ndim - int(axis)

    if axis == 0:
        # transpose data views so columns are treated as rows
        data = data.T
        out_view = out_view.T

    if offset is None:
        # use the first element of each row as the offset
        offset = np.copy(data[:, 0])
    elif np.size(offset) == 1:
        offset = np.reshape(offset, (1,))

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # calculate the moving average
    row_size = data.shape[1]
    row_n = data.shape[0]
    scaling_factors = np.power(1. - alpha, np.arange(row_size + 1, dtype=dtype),
                               dtype=dtype)
    # create a scaled cumulative sum array
    np.multiply(
        data,
        np.multiply(alpha * scaling_factors[-2], np.ones((row_n, 1), dtype=dtype),
                    dtype=dtype)
        / scaling_factors[np.newaxis, :-1],
        dtype=dtype, out=out_view
    )
    np.cumsum(out_view, axis=1, dtype=dtype, out=out_view)
    out_view /= scaling_factors[np.newaxis, -2::-1]

    if not (np.size(offset) == 1 and offset == 0):
        offset = offset.astype(dtype, copy=False)
        # add the offsets to the scaled cumulative sums
        out_view += offset[:, np.newaxis] * scaling_factors[np.newaxis, 1:]

    return out
    
def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out

def get_max_row_size(alpha, dtype=float):
    assert 0. <= alpha < 1.
    # This will return the maximum row size possible on 
    # your platform for the given dtype. I can find no impact on accuracy
    # at this value on my machine.
    # Might not be the optimal value for speed, which is hard to predict
    # due to numpy's optimizations
    # Use np.finfo(dtype).eps if you  are worried about accuracy
    # and want to be extra safe.
    epsilon = np.finfo(dtype).tiny
    # If this produces an OverflowError, make epsilon larger
    return int(np.log(epsilon)/np.log(1-alpha)) + 1