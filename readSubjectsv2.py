import os 
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.signal import hilbert, chirp
import time
import warnings
import pandas as pd
from importlib import reload
from libs import utils,preprocessing
from collections import OrderedDict
import multiprocessing as mp
import re
import json


#train_path_meta = r'C:\Users\MarcWin\Desktop\googleDriveSnycRoot\biomag21'

def get_condition(subject: str) -> str:
    """
    returns:
    -----
    condition: one of 'mci','dementia','control'
    """
    return re.split(r'(\D+)', subject)[1]
   



def temporal_preprocess(X,fs,applyFilter):
    """
    params:
    X, array w/ shape (channels,time_steps)
    
    returns
    ------
    X filtered
    """
    
    if applyFilter:
        # Schirrmeister,2017 und Lawhern,2018 verwenden unteren f_cut von 1 - 4 Hz,je nach Paradigma und FeatureType d.h. ERP 1HZ, Oscillatory 4Hz 
        X = preprocessing.filtering(X,f_cut=0.5,fs=fs,filt_type='high',use0phase=True) 
        # Schirrmeister,2017 und Lawhern,2018 verwenden oberen f_cut von 38 bzw 40 Hz                     
        X = preprocessing.filtering(X,f_cut=200,fs=fs,filt_type='low',use0phase=True)
    #eeg_data-=np.mean(eeg_data,axis=1,keepdims=True)
    return X


def detect_wss_sections(x,length_wss_section,wss_th):
    """
    x np.ndarray num_Channels, num_samples
    Many connectivity measures rely on Wide-Sense-Stationarity (wss). 
    The given MEG data was measured in a Resting State setting with eyes closed, so
    we suppose constant statistical moments. Deviations are results of artificats.
    -----
    return is_wss, array 0 or 1, shape (num_time_steps/length_wss_section) 
    """
    assert (x.shape[1]%length_wss_section)==0
    return np.ones(int(x.shape[1]/length_wss_section))
    
def downsample(x,factor,axis):
    """
    :param x -> array for example (channels,time_steps). x might be 3D if t-f distribution is given.
    :param factor
    :param axis -> you might want to downsample time_steps, so axis=1
    """
    # can be implemented in one line without if else, but takes to long to code
    if axis==0:
        return x[0:len(x):factor]
    elif axis==1:
        return x[:,0:x.shape[1]:factor]
    elif axis==2:
        assert len(x.shape)>=3
        return x[:,:,0:x.shape[2]:factor]
    else:
        raise NotImplementedError

def subjectsId_from_subjects(subjects):
    return [len(subjects['mci']),len(subjects['dementia']),len(subjects['control'])]  

def make_empty_X(num_subjects,fs):
    print(num_subjects,fs)
    return np.empty((num_subjects,160,5*60*fs))#default dtype is np.float64

def condition_to_digit(condition):
    if condition=='mci':
        return 1
    elif condition=='control':
        return 0
    elif condition=='dementia':
        return 2
    else:
        raise ValueError("condition must be mci, control or dementia")

class DataLoader():
    """
    DataLoader was never used. 
    Compared to v2, this loader is more flexibel in order to try different preprocessing.
    """
    def __init__(self,
                train_path_meta:str,
                data_dir:str,
                ch_matching):
        self.train_path_meta = train_path_meta
        self.controlGroup_meta  = os.path.join(self.train_path_meta,'control')
        self.dementiaGroup_meta = os.path.join(self.train_path_meta,'dementia')
        self.mciGroup_meta      = os.path.join(self.train_path_meta,'mci')
        self.data_dir  = data_dir
        self.wss_length = 128
        self.ch_matching = ch_matching
        
    def readData(self,condition,subject_id,**kwargs):
        """
        
        params:
        condition: mci,dementia or control
        subject_id: nbr of subject
        -------
        returns:
        --------
        subjectData, Dictionary
            keys: {'fidpts','data','wss_section','condition','site'}
            
        """
        filename = os.path.join(self.data_dir,condition+'{:03d}'.format(subject_id)+'_raw.npy')
        info = self.load_info(condition,subject_id)
        # split channels    
        X = np.load(filename)
        (num_channels,num_samples) = X.shape
        
       
        is_wss = detect_wss_sections(X,self.wss_length,wss_th='some value') 
        fiducials = utils.get_fiducials(info)
        site = utils.get_site_from_condition_number(condition,subject_id,direc= os.path.join(os.getcwd(),'dataframes','maxwell_and_temporal'))
        
        
        with open(self.ch_matching) as f:
            matching = json.load(f)
            assert len(matching.keys())==1
            key0 = list(matching.keys())[0]
            matching_info = matching[key0].pop('info')
            
            chA = np.array(list(matching[key0].keys())).astype('int')
            chB = np.array(list(matching[key0].values())).astype('int')
            if site=='A':
                X = X[chA]
            else:
                X = X[chB]
        
        subjectData = {}
        subjectData['fiducials']   = fiducials
        subjectData['data']        = X
        subjectData['site']        = site
        subjectData['wss_section'] = is_wss
        subjectData['condition']   = condition
        subjectData['id']          = subject_id
        subjectData['datasum']     = np.sum(X)
        
        return subjectData



    def make_Keras_data(self,subjects: dict,
                        fs:int,
                        use_multiprocessing = False,
                        **readDatakwargs):
        """
        Creates data arrays which can be primarily used with a Keras Model.
        
        params:
        ---------
        subjects: dict 
            keys: 'dementia','mci','control'
            values: list of int representing subject number
        
        train_path: str
        ch_matching: str
            path to json file that matches channels in site A and site B
        Raises:
        ------
        
        Returns:
        --------
        X: MEGArray contains also the label e.g.: 'mci5', 
            shape: (subjects,channels,time_steps)
        y: numpy array, contains the label [0,1,2]
            0 means control
            1 means mci
            2 means dementia
        meta: dictionary
            keys: siteAmeta, siteBmeta, mci{i}, control{j}, dementia{k}
        """
        # todo: If subjects length(keys)<3, and keys in right group. 
        # then add keys with empty list as value
        
        readDatakwargs = {**readDatakwargs,**{'fs':fs}}
        
        
        print(dict(zip(['mci','dementia','control'],subjectsId_from_subjects(subjects))))    
        
        
        siteAmeta = io.loadmat(os.path.join(self.dementiaGroup_meta,'hokuto_dementia{}.mat'.format(1)))# subject from site A
        siteBmeta = io.loadmat(os.path.join(self.dementiaGroup_meta,'hokuto_dementia{}.mat'.format(2)))# subject from site B
        num_subjects = len(subjects['mci'])+len(subjects['dementia'])+len(subjects['control'])    
    
        X            = make_empty_X(num_subjects,fs)
        y            = []
        # merges constant readDataknwargs and variable subject ids.
        readDatadicts = []
        for condition in subjects.keys():
            readDatadicts += [{**readDatakwargs,'condition':condition,'subject_id':sub_id} for sub_id in subjects[condition]]
        print(readDatadicts)
        
        if use_multiprocessing:
            warnings.warn('Multiprocessing has not been tested.')
            with mp.Pool(processes=1) as pool:
                results = pool.starmap(self.readData,readDatadicts)
        else:
            results = [self.readData(**subjectkwargs) for subjectkwargs in readDatadicts]
        
        meta = {'siteAmeta':siteAmeta,'siteBmeta':siteBmeta}
        meta['subjects']=[]
        idxes = []
        y = np.empty(len(results))
        for idx,r in enumerate(results):
            if idx==0: # default dtype of np.empty is float64. We dont want to waste memory 
                # if r.dtype is float32 we set X to float32
                X = X.astype(r['data'].dtype)
            X[idx]=r['data']
            y[idx]=condition_to_digit(r['condition'])
            r.pop('data')
            meta['subjects']+=[r]
        
        return X,y,meta

    def load_info(self,condition,subject_id):
        
        if condition=='dementia':
            info = io.loadmat(os.path.join(self.dementiaGroup_meta,'hokuto_dementia{}.mat'.format(subject_id)))
        elif condition=='mci':
            info = io.loadmat(os.path.join(self.mciGroup_meta,'hokuto_mci{}.mat'.format(subject_id)))
        
        elif condition=='control':
            info = io.loadmat(os.path.join(self.controlGroup_meta,'hokuto_control{}.mat'.format(subject_id))) 
        else:
            raise ValueError("Wrong condition provided")
        return info

#loader = readSubjectsv2.DataLoaderNumpy(readSubjectsv2.train_path_meta,r'E:\2021_Biomag_Dementia_NUMPY\float32\rawfloat32','A_B_graph.json')        
#out = loader.make_Keras_data({'dementia':[1],'mci':[1],'control':[1,23]},fs=128)