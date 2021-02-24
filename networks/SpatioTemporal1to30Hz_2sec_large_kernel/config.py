
import matplotlib
from libs import GeneratorCNN
matplotlib.interactive(False)
from importlib import reload
import gc
from libs import model_zoo
import BaseLineModel as approach
import numpy as np
from importlib import reload
from libs import utils
from libs.utils import Bunch
from libs import Multi_Class_Metrics as mcm
from tensorflow.keras import callbacks


network = model_zoo.SpatialTemporalMultiClass#ConnectivityModel#Channel_Decision_Small#Channel_Decision_Model#ShallowConvNetMultiClass#SpatialTemporalMultiClass#ShallowConvNetMultiClassLinear##SpatialAverageModel# ConnectivityModel#
gc.collect()
use_stable=True
subjects = {'dementia':[],'control':[],'mci':[]}
utils.get_subjects_wrt_site(subjects,30,30,'dementia')
utils.get_subjects_wrt_site(subjects,100,100,'control')
utils.get_subjects_wrt_site(subjects,1,15,'mci')


use_stable='all'
if use_stable=='order0':
    channel_matches = np.load('pareto_opt_matches.npy')
    channel_matches = utils.get_stable_channels(channel_matches,0).T#[:,44:46]
elif use_stable=='all':
    import json
    with open('A_B_bestpositional_hungarian.json') as f:
        match = json.load(f)
    match['matching0'].pop('info')
    channel_matches = np.array([list(match['matching0'].keys()),list(match['matching0'].values())]).astype(int)
elif use_stable is None:
    channel_matches = np.stack([np.arange(160),np.arange(160)])

site_as_label = False
if site_as_label:
  num_classes = 2
else:
  num_classes = 3


sensitivity_mci_dem = mcm.MultiClassRecall(num_classes=num_classes,
                                           pos_ind=[1,2],
                                           average='macro',
                                           name="mci_dem_sensitivity")
specificity_mci_dem = mcm.MultiClassSpecificity(num_classes=num_classes,
                                                pos_ind=[1,2],
                                                average='macro',
                                                name="mci_dem_Specificity")
f1_mci_dem = mcm.MultiClassF1(num_classes=num_classes,
                              pos_ind=[1,2],average='macro',
                              name="mci_dem_F1")
metrics = ['acc', sensitivity_mci_dem, specificity_mci_dem,f1_mci_dem]


callbacks = [callbacks.EarlyStopping(patience=5,
                                     restore_best_weights=True,
                                     monitor="val_mci_dem_F1",
                                     mode='max',
                                     verbose=1)]
frame_size = utils.Bunch(value=2,unit='s')
fs = 256
network_params = {
    'batch_size':32,
    'use_bn':True,
    'do_ratio':0.2,
    'numTrainEpochs':10,
    'optimizer':'adam',
    'frame_size':frame_size,
    'cross_subjects':True,
    'num_classes':num_classes,
    'n_folds':5,
    'use_class_weights':True,
    'workers':1,
    'multiprocessing':False,
    'monitor':"val_mci_dem_F1"
    }

data_params={
             'standardize':'em_astd',#'ema',#'look_at_time',# look_at_time
             'fs':fs,
             'channel_matches':channel_matches,
             'subjects':subjects,
             'test_ratio':0.2,
             'readSubjects_params':{
                 'site_as_label':site_as_label,
                 'l_freq':1,
                 'h_freq':30,
                 'frame_length':utils.time2samples(time_container=frame_size,fs=fs),
                 'bad_samples_path':r'../BioMagData/badsamples1000Hz'
                 },
            }

SA  = np.load('SA.npy')[:,:80]
SB  = np.load('SB.npy')[:,:80]
pSA = np.load('pSA.npy')[:80]
pSB = np.load('pSB.npy')[:80]

# Build augmentation pipeline for Generator
function_params = {}
function_params['draw_random_time_frame']={
   'frame_length':utils.time2samples(fs=data_params['fs'],time_container=network_params['frame_size']),
   'trial_length':5*60*data_params['fs']}
function_params['additive_correlated_noise']={'sigma_noise':5e-2}
function_params['transform_recording_site']={'sensor_inv':[pSA,pSB],
                                             'sensor_fwd':[SA,SB],
                                             'map_to':'random'}
function_params['apply_channel_matches']={'matches':data_params['channel_matches']}

functions=[GeneratorCNN.draw_random_time_frame,
           GeneratorCNN.additive_correlated_noise,
           GeneratorCNN.transform_recording_site,
           GeneratorCNN.apply_channel_matches
           ]

prob = {'apply_channel_matches':1.0,
        'draw_random_time_frame':1.0,
        'transform_recording_site':1.0,
        'additive_correlated_noise':0.5,
        }

train_aug_pipe=GeneratorCNN.Augment_Pipeline(functions=functions,
                                  params=function_params,
                                  p=prob)
func = [
        GeneratorCNN.draw_random_time_frame,
        GeneratorCNN.transform_recording_site,
        GeneratorCNN.apply_channel_matches]
p = {'transform_recording_site':1,'apply_channel_matches':1,'draw_random_time_frame':1}
aug_params = {
    'transform_recording_site':
       {'map_to':'identity',
        'sensor_inv':[pSA,pSB],
        'sensor_fwd':[SA,SB]},
    'apply_channel_matches':
        {'matches':data_params['channel_matches']},
    'draw_random_time_frame':
        {'frame_length':utils.time2samples(fs=data_params['fs'],time_container=network_params['frame_size']),
        'trial_length':5*60*data_params['fs']}
        }
valid_pipe = GeneratorCNN.Augment_Pipeline(func,aug_params,p)