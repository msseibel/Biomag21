import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal,fftpack
import itertools
from importlib import reload
import gc
from sklearn.model_selection import GroupKFold,train_test_split,KFold, StratifiedKFold
from collections import OrderedDict
import pandas as pd
from tensorflow import keras
import readSubjectsfif as readSubjects
import json
from libs import model_zoo
from libs import utils,GeneratorCNN,visualisations, calc_scores
from tensorflow.keras import callbacks
from sklearn.utils import class_weight
import warnings 


def assembleLoad(self,condition_subjects_dict,**kwargs):
    """
    param: condition_subjects_dict e.g. {'mci':[1,2],'control':np.arange(10),'dementia':[1,2,3]}
    """
    loader = readSubjects.DataLoader(self.train_path_info,
        self.dataDir,self.static_data_params['channel_matches'],**kwargs)
    out = loader.make_Keras_data(condition_subjects_dict,fs=self.static_data_params['fs'])
    data = {}
    data['X'] = np.transpose(out[0],[0,2,1])
    data['Y'] = out[1]
    meta      = out[2]
    return data,meta

networkname = input('Folder where the Training is saved: ')
with open(os.path.join('networks',networkname,'test_subjects.json'),'r') as test_subjects:
    json.loads(test_subjects)
    
