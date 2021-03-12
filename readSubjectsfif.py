import mne
import numpy as np
import os 
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
from libs import utils
from tqdm import tqdm
from importlib import reload  # Not needed in Python 2

#train_path_meta = r'C:\Users\MarcWin\Desktop\googleDriveSnycRoot\biomag21'

def get_condition(subject: str) -> str:
    """
    returns:
    -----
    condition: one of 'mci','dementia','control'
    """
    return re.split(r'(\D+)', subject)[1]
   
def get_bad_times(condition,subject,signal_length=76800,frame_length=256,bad_samples_path=''):
    from skimage import transform
    bads = utils.load_bad_samples(condition,subject,path=bad_samples_path)
    out = transform.resize(bads,[160,signal_length],anti_aliasing=False,preserve_range=True)
    bads_indices = np.where(np.sum(out,axis=0)>30)[0]
    earlier     = np.clip(bads_indices-frame_length,0,signal_length-frame_length)
    merged_bads = np.unique(earlier)
    return merged_bads

def get_bad_times_one_hot(condition,subject,signal_length=76800,bad_samples_path="."):
    merged_bads = get_bad_times(condition,subject,signal_length,bad_samples_path)
    really_bad = np.zeros(signal_length)
    really_bad[merged_bads]=1
    return really_bad

def resample_goods(merged_bads,signal_length=76800,frame_length=256):
    """
    We want equal length arrays with good samples
    """
    if len(merged_bads)==0:
        return np.arange(signal_length-frame_length)
    spacing = (signal_length-frame_length)//len(merged_bads)
    goods = np.concatenate([np.setdiff1d(np.arange(signal_length-frame_length),merged_bads),
                    (np.arange(signal_length-frame_length)[::spacing])[:len(merged_bads)]])
    return goods

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
    

def subjectsId_from_subjects(subjects):
    return [len(subjects['mci']),len(subjects['dementia']),len(subjects['control'])]  

def make_empty_X(num_subjects,fs,num_bands=None):
    #print(num_subjects,fs)
    if num_bands is not None:
        return np.empty((num_subjects,160,num_bands,5*60*fs))#default dtype is np.float64
    else:
        return np.empty((num_subjects,160,5*60*fs))#default dtype is np.float64


# 
class DataLoader():
    """
    todo: rename to DataLoaderFIF and build baseclase DataLoader that also has the Numpy loader
    DataLoader was never used. 
    Compared to v2, this loader is more flexibel in order to try different preprocessing.
    """
    def __init__(self,
                train_path_meta:str,
                data_dir:str,
                ch_matching=None,
                site_as_label=False,**kwargs):
        """
        train_path_meta: 
        data_dir: 
        ch_matching is ignored
        """
        self.train_path_meta = train_path_meta
        self.controlGroup_meta  = os.path.join(self.train_path_meta,'control')
        self.dementiaGroup_meta = os.path.join(self.train_path_meta,'dementia')
        self.mciGroup_meta      = os.path.join(self.train_path_meta,'mci')
        self.data_dir  = data_dir
        self.site_as_label=site_as_label
        if site_as_label:
            warnings.warn('site as label')

    def readData(self,file,**kwargs):
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
        filename = os.path.join(self.data_dir,file)
        condition,subject_id = utils.parse_filename(filename)
        
        # split channels
        use_filter = False
        if 'l_freq' in kwargs.keys():
            l_freq = kwargs['l_freq']
            use_filter = True
        else:
            l_freq = 0
            
        if 'h_freq' in kwargs.keys():
            h_freq = kwargs['h_freq']
            use_filter = True
        else:
            h_freq = 100
        
        if 'frame_length' in kwargs.keys():
            frame_length = kwargs['frame_length']
        else:
            frame_length = 256
            
        if 'num_channels' in kwargs.keys():
            num_channels = kwargs['num_channels']
        else:
            # for KIT/Yokogawa
            num_channels = 160
        #if 'num_samples' in kwargs.keys():
        #    num_samples = kwargs['num_samples']
        #else:
        #    # for KIT/Yokogawa at 256Hz with 5 minute recording
        #    num_samples = 76800
        if 'fs' in kwargs.keys():
            fs = kwargs['fs']
        if 'utility_data' in kwargs.keys():
            utility_data_path = kwargs['utility_data']
        else:
            utility_data_path = "."
        if 'verbose' in kwargs.keys():
            verbose = kwargs['verbose']
        else: 
            verbose = 0
        
        X = mne.io.read_raw_fif(filename,verbose=verbose)
        if fs is not None and X.info['sfreq']!=fs:
            X.load_data()
            X.resample(fs)
                
        # https://mne.tools/stable/auto_examples/time_frequency/plot_time_frequency_global_field_power.html?highlight=bands
        if hasattr(l_freq, '__iter__'): #implies use_filter
            assert len(l_freq)==len(h_freq)
            #logging.info('Extracting multiple frequency bands. This can take a long time.')
            bands = np.zeros((num_channels,len(l_freq),X.n_times))
            for i,(lf,hf) in enumerate(zip(l_freq,h_freq)):
                Xcopy = X.copy()
                Xcopy.load_data()
                #logging.info('Frequency Band Extraction uses a small transition bandwidth of .5 Hz.')
                #logging.info('Using non causal filter. This is unwanted for evoked response detection.')
                Xcopy.filter(l_freq=lf,h_freq=hf,verbose=verbose,l_trans_bandwidth=.5,h_trans_bandwidth=.5)
                bands[:,i] = Xcopy.get_data()
            num_samples = X.n_times
            
            X = mne.time_frequency.AverageTFR(Xcopy.info,bands,
                times=np.arange(0,num_samples)/Xcopy.info['sfreq'],
                freqs=np.stack([l_freq,h_freq],axis=1),
                nave=1)
        elif use_filter:
            #print('Load and Filter: ',condition, ' ', subject_id)
            X.load_data()
            X.filter(l_freq=l_freq,h_freq=h_freq,verbose=verbose)
     
        
        if hasattr(X,'n_times'):
            signal_length = X.n_times
        else:
            signal_length = X.times.shape
        
        if 'bad_samples_path' in kwargs.keys():
            bad_samples_path = kwargs['bad_samples_path']
            merged_bads = get_bad_times(condition,subject_id,signal_length,bad_samples_path=bad_samples_path,
                frame_length=frame_length)    
            goods = resample_goods(merged_bads,signal_length=signal_length,frame_length=frame_length)
        else:
            goods = None#==np.arange(len(signal_length))
        site = utils.get_site_from_condition_number(condition,
                    subject_id,
                    direc= utility_data_path)
        subjectData = {}
        subjectData['data']        = X
        subjectData['site']        = site
        #subjectData['artifacts']   = is_wss
        subjectData['condition']   = condition
        subjectData['filename'] = file
        subjectData['id'] = subject_id
        subjectData['good_samples'] = goods 
        return subjectData


    
    def make_Keras_data(self,subjects: dict,
                        fs:int,
                        use_multiprocessing = False,
                        preload=True,
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
        X: (MEGArray | list of mne.RawArray)
            if preload=True MEGArray (WIP might change to plain numpy array)
            contains also the label e.g.: 'mci5', shape: (subjects,channels,time_steps),
            if preload=False: returns list of ,mne.RawArray
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
        
        conditions = ['mci','dementia','control']
        print(dict(zip(conditions,subjectsId_from_subjects(subjects))))    
        
        
        siteAmeta = None#io.loadmat(os.path.join(self.dementiaGroup_meta,'hokuto_dementia{}.mat'.format(1)))# subject from site A
        siteBmeta = None#io.loadmat(os.path.join(self.dementiaGroup_meta,'hokuto_dementia{}.mat'.format(2)))# subject from site B
        num_subjects = len(subjects['mci'])+len(subjects['dementia'])+len(subjects['control'])    
    

        # compare input subjects with files in data_dir
        allfiles = [f for f in os.listdir(self.data_dir) if f.endswith('.fif')]
        files = []
        for f in allfiles:
            condition,subid = utils.parse_filename(f)
            if subid in subjects[condition]:
                files+=[f]
        if len(files)==0:
            raise ValueError('Cant find data files')
        # merges constant readDataknwargs and variable subject ids.
        readDatadicts = []        
        for f in files:
            readDatadicts += [{**readDatakwargs,'file':f}]

        if preload:
            if 'l_freq' in readDatakwargs.keys() and hasattr(readDatakwargs['l_freq'],'__iter__'):
                l_freq = readDatakwargs['l_freq']
                X = make_empty_X(num_subjects,fs,num_bands=len(l_freq)).astype(np.float32)
            else: 
                X = make_empty_X(num_subjects,fs).astype(np.float32)
        else:
            X = []
        print('')    
        if use_multiprocessing:
            warnings.warn('Multiprocessing has not been tested.')
            with mp.Pool(processes=2) as pool:
                results = pool.starmap(self.readData,readDatadicts)
        else:
            results = [self.readData(**subjectkwargs) for subjectkwargs in tqdm(readDatadicts,position=0,leave=True)]
        
        

        meta = {'siteAmeta':siteAmeta,'siteBmeta':siteBmeta}
        meta['subjects']=[]
        idxes = []
        y = np.empty(len(results)).astype(np.int8)
        for idx,r in enumerate(results):
            # todo: add check: r['data'] is mne.Raw instance
            # also meta must be filled
            if preload:
                if hasattr(r['data'],'get_data'):
                    X[idx]=r['data'].get_data()*1e15
                else:
                    X[idx]=r['data'].data*1e15
            else:
                X+=[r['data']]
            if self.site_as_label:
                y[idx] = r['site']=='B'
            else:
                y[idx]=utils.condition_to_digit(r['condition'])
            r.pop('data')
            meta['subjects']+=[r]
        
        # check subjects delivered:
        assert len(meta['subjects'])==num_subjects,('Is: {} should: {}'.format(len(meta['subjects']),num_subjects))   
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