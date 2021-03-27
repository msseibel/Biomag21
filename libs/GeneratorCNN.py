from tensorflow import keras
import numpy as np
import gc
from libs import preprocessing, utils
import pandas as pd
from random import seed
from random import random
from typing import Union
import warnings

def popartifact(time_axis,height,tau,when):    
    artifact = height*np.exp(-(time_axis-when)/tau)*(np.sign(time_axis-when)+1)/2
    return artifact

def get_sphara_operator(points3D, trilist):    
    points3D = np.squeeze(points3D)
    trilist = np.squeeze(trilist)
    # fit:
    mesh_eeg = tm.TriMesh(trilist, points3D)
    sphara_basis = sb.SpharaBasis(mesh_eeg, 'fem')
    basis_functions, natural_frequencies = sphara_basis.basis()
    #print('eigenvalues: \n',natural_frequencies)
    sphara_transform_fem = st.SpharaTransform(mesh_eeg, 'fem')
    return sphara_transform_fem
    
def apply_sphara_operator(eegdata,sphara_operator):
    """
    eegdata: time_steps,channels
    """
    sphara_data = sphara_operator.analysis(np.transpose(eegdata)) 
    return sphara_data
    
def transform_sphara(X,points3D,trilist):
    """
    X: time_steps,channels
    """
    sphara_operator = get_sphara_operator(points3D,trilist)
    return apply_sphara_operator(X,sphara_operator)
    
def get_TriMesh(points3D,trilist):
    points3D = np.squeeze(points3D)
    trilist = np.squeeze(trilist)
    # create an instance of the TriMesh class    
    return tm.TriMesh(trilist, points3D)

def get_sphara_filter(mesh_eeg,filter_spec):
    # create a filter,
    # computational effort: only simple if else conditions which mask values
    sphara_filter_fem = sf.SpharaFilter(mesh_eeg, mode='fem',
                                    specification=filter_spec)
    return sphara_filter_fem

    
class Augmentation_Function(object):
    def __init__(self,head_params=None,**kwargs):
        self.states = {'epoch':-1,'index':-1,'head_params':head_params}
    def __call__(self,*args,**kwargs):
        pass
    def _update_state(self):
        self.states['index']+=1
    def on_epoch_end(self,**kwargs):
        self.states['epoch']+=1
        self.states['index'] = 0
            
# test construct, No specific use
class MyIterator(Augmentation_Function):
    def __init__(self,**kwargs):
        super().__init__()        
    
    def __call__(self,Xtmp,**kwargs):
        return Xtmp
        
    def on_epoch_end(self):
        self.states['head_params']='NewValue'
        super().on_epoch_end()
        
class meanstd(Augmentation_Function):
    def __init__(self,**kwargs):
        super().__init__()
        
    def __call__(self,Xtmp,**kwargs):
        Xtmp = Xtmp-np.mean(Xtmp,axis=-1,keepdims=True)
        return Xtmp/np.std(Xtmp,axis=-1,keepdims=True)


class additive_correlated_noise(Augmentation_Function):
    def __init__(self,sigma_noise=0.01):
        self.sigma_noise = sigma_noise
        super().__init__()
        
    def __call__(self,Xbatch,**kwargs):
        """
        
        all channels get same noise, as it is often in realistic eeg measurements.
        One may configure this method in order to add noise just to a subset of eeg-channels.
        -----------
        params: Xbatch: [batch_size,time_steps,channels] 
        """
        correlated_noise = np.reshape(np.random.randn(Xbatch.shape[1])*self.sigma_noise,[1,Xbatch.shape[1],1])
        Xbatch = Xbatch+correlated_noise
        return Xbatch, kwargs 
        
#f,t,Sxx = filter_sphara(xtmp, points3Dtmp, trilisttmp,patient,sphara_filter_spec)
class apply_sphara_filter(Augmentation_Function):
    def __init__(self,specification_limits):
        super().__init__()
        self.specification_limits = specification_limits
    
    def __call__(self,batch,**kwargs):
        """
        param: batch (batch_size,time_points,channels)
        """
        
        # filter with a random low pass degree
        sphara_filter = get_sphara_filter(self.TriMesh,
                            np.random.randint(
                                self.specification_limits[0],
                                self.specification_limits[1])
                                )
        eegdata = np.reshape(batch,[batch.shape[0]*batch.shape[1],batch.shape[2]])
        # input and output shape of filter: time_steps,channels
        sphara_filt_eegdata = sphara_filter.filter(eegdata)
        #if (self.states['epoch']+self.states['index'])==0:
        #    print('shape sphara filter: ', sphara_filt_eegdata.shape)
        #print('shape sphara filter: ', sphara_filt_eegdata.shape)
        sphara_filt_eegdata = np.reshape(sphara_filt_eegdata,[batch.shape[0],batch.shape[1],batch.shape[2]])
        return sphara_filt_eegdata,kwargs
    
    def on_epoch_end(self,sensorPos,triList,**kwargs):
        """
        param sensorPos: array(subjects,N_sensors,3)
        param triList:  array(subjects,..)
        """
        ### Do specific stuff.
        # take random head:
        head = np.random.randint(len(sensorPos))
        self.TriMesh = get_TriMesh(sensorPos[head],triList[head])
        ### Do the default stuff.
        super().on_epoch_end()
        
class draw_random_time_frame(Augmentation_Function):
    
    def __init__(self,frame_length: int,trial_length: int):
        """
        frame_length:
        trial_length:
        """
        self.frame_length = int(frame_length)
        self.trial_length = int(trial_length)
        assert trial_length>frame_length
        super().__init__()
        
    def __call__(self,Xbatch, good_samples=None,**kwargs):
        """
        Xbatch.shape: (batch_size,time_steps,channels)
        good_samples: (batch_size,time_steps-frame_length)
        see readSubjectsfif for how to create good_samples
        """
        trial_length = Xbatch.shape[1]
        batch_size = len(Xbatch)
        start_time_point        = np.random.randint(0,trial_length-self.frame_length,size=batch_size)
        if good_samples is not None:
            indices = np.random.randint(good_samples.shape[1],size=batch_size)
            start_time_point = good_samples[np.arange(batch_size),indices]
            #print(start_time_point+self.frame_length)
        start_time_points_range = [np.arange(sp,sp+self.frame_length) for sp in start_time_point]
        samplesRepeated         = np.repeat(np.arange(batch_size).reshape(batch_size,1),self.frame_length,axis=1)
        return Xbatch[samplesRepeated,np.array(start_time_points_range)],kwargs

class draw_continuous_time_frame(Augmentation_Function):
    """
    Start from sample 0 and end when start_time_point+frame_length == time_steps
    That means a grid of start time points must be defined and a function to samples from it.
    """
    def __init__(self,overlap: int,frame_length: int):
        self.start_time_point = 0
        self.frame_length = int(frame_length)
        self.increment = int(frame_length)-int(overlap)
        super().__init__()

    def __call__(self,Xbatch, good_samples=None,**kwargs):
        """
        Xbatch.shape: (samples,time_steps,channels)
        """
        batch_size = Xbatch.shape[0]
        samplesRepeated = np.repeat(np.arange(batch_size).reshape(batch_size,1),self.frame_length,axis=1)
        
        batch_inc = batch_size*self.increment
        
        
        if self.frame_length+self.start_time_point-batch_inc>Xbatch.shape[1]:
            raise IndexError('Snapshot exceeds signal length.')
        
        if good_samples is not None:
            indices = np.arange(self.start_time_point-batch_inc,self.start_time_point,self.increment)
            start_time_points = good_samples[np.arange(batch_size),indices]
        else:    
            start_time_points = np.arange(self.start_time_point-batch_inc,self.start_time_point,self.increment)
        start_time_points_range = [np.arange(sp,sp+self.frame_length) for sp in start_time_points]
        
        Xb = Xbatch[samplesRepeated,start_time_points_range].copy()
        self.start_time_point+=batch_inc
        return Xb,kwargs
    
    def on_epoch_end(self):
        self.start_time_point = 0

class apply_channel_matches(Augmentation_Function):
    """
    uses fancy indexing and permutes channels:
    subset channel selection is also possible 
    by passing indexes smaller then max channel index 
    (just use less indices then for full channel selection) 
    e.g. 
    data = np.arange(2*1*5).reshape(2,1,5)#subjects,time,channels
    np.transpose(data[[[0],[1]],:,[[0,1,2,3,4],[4,0,1,2,3]]],[0,2,1])
    
    array([[[0],
        [1],
        [2],
        [3],
        [4]],

       [[9],
        [5],
        [6],
        [7],
        [8]]])
    
    Example:
    >>> data = np.arange(2*2*5).reshape(2,2,5)
    >>> matches
    array([[0, 1, 2, 3, 4],
           [4, 0, 1, 2, 3]])    
    >>> sites = np.array([0,1])
    >>> matcher = GeneratorCNN.apply_channel_matches(matches)
    >>> matcher(data,sites)
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9]],

           [[14, 10, 11, 12, 13],
            [19, 15, 16, 17, 18]]])
    """    
    def __init__(self,matches):
        if len(matches.shape)==2:
            if matches.shape[0]==2:# check form of array is [[siteA,siteB],num_channels]
                matches = np.expand_dims(matches,axis=0)
        assert len(matches.shape)== 3
        self.matches = matches
        super().__init__()
    
    def __call__(self,Xbatch,subject_sites,**kwargs):
        """
        Xbatch: subjects, time, channels
        subject_sites: array len(Xbatch) with entries 0,1
            where 0 corresponds to site A and 1 to site B.
        """    
        match = np.random.randint(len(self.matches))
        matches = self.matches[match,subject_sites]
        Xbatch = np.transpose(Xbatch[
            np.arange(len(Xbatch)).reshape(len(Xbatch),1),
            :,
            matches
            ],[0,2,1])            
        kwargs['subject_sites']=subject_sites
        return Xbatch,kwargs

class transform_recording_site(Augmentation_Function):
    """
    MEG data can be transformed between recording sites via e.g. the Leadfield method (Knösche 2002) or SSS (Taulu 2004).
    The leadfield methods requires to compute the leadfield matrices by modelling the source space.
    
    For SSS the python MNE library can be used:
    SA,pSA,reg_momentsA,n_use_inA = mne.preprocessing.maxwell.info_maxwell_basis(rawA.info,origin=origin,regularize=None)
    SB,pSB,reg_momentsB,n_use_inB = mne.preprocessing.maxwell.info_maxwell_basis(rawB.info,origin=origin,regularize=None)
        
    sensor_inv: 
        list of matrices transforming data from sensor space into device independent space
        in case of SSS it is a list of the pseudo inverse of S_in: pS_inA, pS_inB,...
    sensor_fwd: list of matrices transforming data from device independent space into sensor space
        in case of SSS it is a list S_in: S_inA, S_inB
    
   
    TLDR: The method computes:
        dataBtoA = SA @ pSB @ dataB[all_channels,any_samples]
        
    If len(sensor_inv)>2 then map_to will be evaluated. 
    If map_to is 'random', then data from MEG device_j will randomly mapped to any device_i, j!=i
    elseif map_to=='identity' then projectors will be applied s.t. data is mapped to the original device.
    """
    
    def __init__(self,sensor_inv,sensor_fwd,map_to='random'):
        self.map_to = map_to
        self.sensor_inv = np.array(sensor_inv)
        self.sensor_fwd = np.array(sensor_fwd)
        
        if self.map_to=='inverse':
            assert len(self.sensor_inv)==2
        
        assert len(self.sensor_inv)==len(self.sensor_fwd)
        self.num_devices = len(self.sensor_inv)
        num_channels = self.sensor_inv.shape[-1]
        # num_devices,num_devices, num_channels, num_channels
        
        # Device Identity is on the diagonal axis 
        self.projections = np.zeros((self.num_devices,self.num_devices,num_channels,num_channels))
        for i,pS in enumerate(self.sensor_inv): 
            for j,S in enumerate(self.sensor_fwd):
                self.projections[i,j] = S@pS        
        super().__init__()
    
    def __call__(self,Xbatch,subject_sites, target_sites=None,**kwargs):
        """
        Xbatch: subjects, time, channels
        subject_sites: array len(Xbatch) with entries 0,1
            where 0 corresponds to site A and 1 to site B.
        target_base list,|ndarray: 
            idxs of site to where subjects shall be transformed
        
        The method swaps the site and thereby the ordering of channels.
        """
        batch_size = len(Xbatch)

        subject_sites = subject_sites.astype(int)
        if target_sites is None and self.map_to=='random':
            target_sites = np.random.randint(self.num_devices,size=batch_size)
        elif self.map_to=='identity':
            target_sites = subject_sites
        elif self.map_to=='inverse':
            target_sites = (~(subject_sites).astype(np.bool)).astype(int)
        elif self.map_to=='B':
            target_sites = np.ones(len(subject_sites)).astype(int)
        elif self.map_to=='A':
            target_sites = np.zeros(len(subject_sites)).astype(int)
        else:
            target_sites = np.array(target_sites)

        if self.num_devices==2:
            #print(subject_sites,target_sites)
            sensor_inv = self.sensor_inv[subject_sites]
            sensor_fwd = self.sensor_fwd[target_sites]
        else:
            raise NotImplementedError('Currently only 2 different devices are supported')
                
        # batch_size,num_channels, num_channels
        projections = self.projections[subject_sites,target_sites]

        # do it iterative. It can be done with kronecker product but that is hard to write and understand
        # it might also take a lot of memory
        Xbatch = np.transpose(Xbatch,[0,2,1])
        for k in range(batch_size):
            Xbatch[k]=projections[k]@Xbatch[k]
        # back swap time and channel axis
        kwargs['subject_sites'] = target_sites
        #kwargs['target_sites'] =  target_sites
        return np.transpose(Xbatch,[0,2,1]),kwargs


class draw_time_frame(Augmentation_Function):
    """
    Defining a start time point is not easy. i.e. 
    Do we want to check each possible time point for each subject...?
    
    That means a grid of start time points must be defined and a function to samples from it.
    """
    def __init__(self):
        super().__init__()

    def __call__(self,Xbatch,start_time_point,frame_length,**kwargs):
        """
        Xbatch: subjects,time,channels
        """
        return Xbatch[:,start_time_point:frame_length+start_time_point]
    
class Augment_Pipeline():
    def __init__(self,functions=[],params={},p={},head_params=None,debug=False):
        """
        functions: list of callable. Must have attribute __name__.
        params:    multi level dictionary with first keys being the names of the callables
        p: dict, with function names as keys, like params. But the values are probabilities for applying the function 
        """
        
        assert len(params)==len(functions),"params and functions dont have same length."
        assert len(p)==len(functions),"params and functions dont have same length."
        self.debug=debug
        self.functions = functions
        self.params    = params
        self.p = p
        self.initial_states = {'epoch':0,'index':0}
        self.states = {'epoch':0,'index':0,'head_params':head_params}
        
        # Some functions might depend on the previous function:
        # E.g. apply_channel_matches needs different input when transform_recording_site is also used
        # For these cases `Augment Pipeline` provides a memory.
        self.changing_function_params = {'subject_sites':None}
        self.initialized_functions = []
        for func in self.functions:
            params_func = self.params[func.__name__]
            self.initialized_functions+=[func(**params_func)]
            
    def __call__(self,batch,**kwargs):
        for func in self.initialized_functions:
            p_func = self.p[func.__class__.__name__]
            if random()>p_func:
                continue
            if self.debug:
                print('call: ',func.__class__.__name__)
                print(kwargs)
            batch,kwargs = func(batch,**kwargs)    
        self.states['index']+=1
        return batch
    
    def on_epoch_end(self,data):
        """
        Some augmenter-functions require an update after each epoch.
        """
        self.states['epoch']+=1
        for func in self.initialized_functions:
            #print(func.__class__.__name__)
            func.on_epoch_end(**data)
        
    def set_state(self,state,new_value):
        """Different Interface."""
        self.states[state] = new_value
        
    def reset_state(self):
        self.states = self.initial_states
        pass        
"""
# Example:
dd = {}
dd['meanstd']={}
dd['apply_sphara_filter']={'sphara_filter':get_sphara_filter(sensorPos, triList,filter_spec)}
functions=[meanstd,apply_sphara_filter]
asd=GeneratorCNN.Augment_Pipeline(functions=functions,params=dd)
"""
def categorical2onehot(y,num_classes=None):
    """
    y: (samples)
    """
    if not num_classes:
        classes = np.unique(y).astype('int')
        # assert that labels are integers or that we can cast them to int
        assert np.sum(classes-np.unique(y))==0 
        num_classes = len(classes)
    else:
        # It is conceivable, that your problem has 10 classes, 
        # but a given y only contains 8
        raise NotImplementedError        
    y_onehot = np.zeros((y.shape[0],num_classes))
    for  c in classes:
        ar = np.zeros(num_classes)
        ar[c]=1
        y_onehot[y==c]=ar
    return y_onehot
    
def onehot2categorical(y):
    """
    y: (samples,num_classes)
    """
    return np.argmax(y,axis=-1)

class SlicingGeneratorAugment(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self,static_data_params,network_params,augmenter=None,shuffle=True,debug=False):
        """
        From each subjects show an average number of k frames to the network
        in each epoch.
        
        Initialization
        :param self:
        :param DataFrame:
        :param batch_size: defaults 32
        :param dim:
        :param n_channels: defaults 1
        :param n_classes: defaults 2
        :param shuffle:
        :param output_layers:
        :param input_layers:
        :param debug_mode:
        :return:
        """
        self.network_params = network_params
        self.debug = debug
        if static_data_params['channel_matches'] is None:
            self.matches = np.expand_dims(np.stack([np.arange(160),np.arange(160)],axis=0),axis=0)
        else:
            self.matches = static_data_params['channel_matches']
            warnings.warn('Make sure that the recording site is provided for each subject.')
        
        frame_length = utils.time2samples(**{'fs':static_data_params['fs'],
            'time_container':   utils.Bunch(**network_params['frame_size'])})
        
        if augmenter is None:
            func = [apply_channel_matches,draw_random_time_frame]
            p = {'apply_channel_matches':1,'draw_random_time_frame':1}
            aug_params = {
                'apply_channel_matches':
                    {'matches':self.matches},
                'draw_random_time_frame':
                    {'frame_length':frame_length,
                    'trial_length':5*60*static_data_params['fs']}
                    }
            augmenter = Augment_Pipeline(func,aug_params,p)
        
        if 'overlap' in augmenter.params.keys():
            self.overlap = augmenter.params['overlap']
        else:
            self.overlap = 0
        
        
        randomSeed = 86754231
        # myRandom = Random(randomSeed)
        np.random.seed(randomSeed)
        
        
        # [params that define what we want to do with data]
        # time of one crop for training network_params['frame_size'] in [ms]
        self.frame_length = frame_length
        self.batch_size   = network_params['batch_size']
        self.shuffle = shuffle
        self.augmenter = augmenter
        
        
    def setData(self, data, subjects,meta):
        """
        data={'X':np.arange(32*5*60*8).reshape(32,5*60,8),'Y':np.random.randint(2,size=32)}
        meta={'subjects':[{'site':np.random.choice(['A','B'])} for k in range(32)]}
        
        assigns data to generator object.
        This should not create a copy s.t. no additional memory is required.
        Data assignment is explained here:
        https://stackoverflow.com/questions/986006/how-do-i-pass-a-variable-by-reference
        and here:
        https://docs.python.org/3/faq/programming.html#how-do-i-write-a-function-with-output-parameters-call-by-reference
        """
        assert len(data['X'])==len(meta['subjects'])
        # these are the subjects that are present during training or test mode
        self.subjects = subjects
        
        #self.batch_size = np.min([self.subjects,self.batch_size])
        
        
        self.meta  = np.array(meta['subjects'])
        self.conditions       = np.array([sub_meta['condition'] for sub_meta in self.meta])
        self.subjects_id      = np.array([sub_meta['id'] for sub_meta in self.meta])
        subject_names = np.array(["{}{:03d}".format(cond,sid) for cond,sid in zip(self.conditions,self.subjects_id)])
        
        self.X         = data['X']
        self.Y         = data['Y']
        good_samples = np.array([sub_meta['good_samples'] for sub_meta in self.meta])
        sites        = np.array([int(sub_meta['site']=='B') for sub_meta in self.meta])
        if len(np.unique(self.Y))>2:
            self.Y = categorical2onehot(self.Y)
        
        
        self.mode_sites = sites[self.subjects]
        self.mode_Y     = self.Y[self.subjects]
        self.mode_good_samples = good_samples[self.subjects]
        mode_subject_names = subject_names[self.subjects]
        
        self.rng = np.random.default_rng()
        self.samples      = self.X.shape[1]
        self.numChannels  = self.X.shape[2]
        # contains only the subjects as specified by subjects index
        # len(np.unique(subject_names))
        self.num_subjects = len(subjects)  this is only valid when data was not pre augmented

        #--------------------------------------------------------------
        #--------------------------------------------------------------
        #--------------------------------------------------------------
        # We want to show samples from both sites equally often
        site_types = np.unique(sites)
        num_sites = len(site_types)
        # todo: add good criteria when to use weighted site sampling, but outside of Generator
        if 'site_sampling' in self.network_params.keys() and self.network_params['site_sampling'] and len(subjects)>1 and np.sum(self.mode_sites==1)!=0:
            # 0 repr site A and 1 repr site B
            ratio_AtoB = np.sum(self.mode_sites==0)/np.sum(self.mode_sites==1)
            self.mode_probs = np.ones_like(self.subjects)
            self.mode_probs[np.where(self.mode_sites==1)] = ratio_AtoB
            self.mode_probs = self.mode_probs / np.sum(self.mode_probs)
        else:
            self.mode_probs = np.ones_like(self.subjects)
            self.mode_probs = self.mode_probs / np.sum(self.mode_probs)
        #--------------------------------------------------------------
        #--------------------------------------------------------------
        #--------------------------------------------------------------
        
        
        
        if self.num_subjects<self.batch_size:
            self.replace=True
        else:
            self.replace=False
        gc.collect()
        self.not_calculated = True
        self.__len = self.__len__()
        #print('Number of batches: ',self.__len)
        self.on_epoch_end()
    
    def unset_data(self):
        del self.X,self.Y

    def resetIndexes(self):
        self.indexes = np.arange(self.__len * self.batch_size)

    def __len__(self):
        """Denotes the number of batches/steps per epoch"""
        # number_of_frames = len(np.concatenate(self.slices.flatten()))  # unpack slice numbers
        if self.not_calculated:
            # train with k frames from each subjects ...
            # k is the minimum number of snapshots we would get if 
            # the signal was separated into non-overlapping snapshots.
            # During training the snapshots are random.
            # This must be noted in the seminar.
            if self.overlap==0:
                k = self.samples//self.frame_length
            else:
                jump = self.frame_length-self.overlap
                k = (self.samples-self.frame_length)//jump+1
            # these k frames are provided in batches
            # The number of batches is returned
            self.not_calculated = False
            return int(np.floor(self.num_subjects * k // self.batch_size))
        else:
            return self.__len

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if not hasattr(self,'indexes'):
            self.resetIndexes()
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        # Set epoch specific augmentation params.
        data={}
        self.augmenter.on_epoch_end(data)

    def __getitem__(self, index):
        """Generate one batch of data
        """
        # show subjects from site B more often since there are less subjects from this side
        batch_subjects = self.rng.choice(self.num_subjects,
                                         size=self.batch_size,
                                         replace=self.replace,
                                        p=self.mode_probs)
        batch_sites = self.mode_sites[batch_subjects]
        batch_good_samples = self.mode_good_samples[batch_subjects]
        if self.debug:
            print(batch_good_samples.shape)
        # you can add params for specific functions. They will be selected automatic.
        # self.subjects selects train or valid subjects respectively
        
        params = {
             'subject_sites': batch_sites,
             'good_samples': batch_good_samples,
            }
            
        Xbatch = self.augmenter(self.X[self.subjects][batch_subjects],**params)
        if self.debug:
            print('batch index: ',index)
            print('subjects in batch: ',self.subjects[batch_subjects])
            print('with site: ',batch_sites)
            print('label: ',self.mode_Y[batch_subjects])
            print('condition: ',self.conditions[self.subjects][batch_subjects])
            print('sub id: ',self.subjects_id[self.subjects][batch_subjects])
            print('params: ',params)
            print('batch shape: ', Xbatch.shape)
            
        #if 'num_outputs' in self.network_params:
        
        return np.expand_dims(Xbatch,axis=-1),self.mode_Y[batch_subjects]
#else
#    return (np.expand_dims(Xbatch,axis=-1),np.repeat(
#        np.expand_dims(self.Y[self.subjects][batch_subjects],axis=1)
#        self.network_params['num_outputs'],axis=1)