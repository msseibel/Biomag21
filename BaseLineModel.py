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
import shutil
import pickle
from libs import Multi_Class_Metrics as mcm
from tqdm import tqdm
#import jsonpickle #!pip install jsonpickle

LOADED = False
"""
data_params = {
    'fs':128,
    'standardize':'look_at_all_axis'}


network_params = {
    'earlyStopPatience':5,
    'batch_size':32,
    'N_FOLDS':1,
    'useRandomLabels':False,
    'use_bn':True,
    'do_ratio':0.01,
    'numTrainEpochs':10,
    'num_classes':3,
    'optimizer':'adam',
    'frame_size':utils.UnitData(128,unit=1),
    'cross_subjects':True}
"""
#import BaseLineModel as approach
#network = model_zoo.ShallowConvNet
#trainer = approach.RunTraining()
#trainer.start_main(network_params,data_params,network=network)
class RunTraining():
    """
    Why Using a class?: 
    I want to load data just once, but rerun training with different settings.
    I admit there are more elegant ways, but this was the first which came to mind.
    """
    def __init__(self,dataDir=r'E:\2021_Biomag_Dementia_NUMPY\float32\rawfloat32',
            train_path_info=r'info',static_data_params=None,network_params=None,completeDir=None):
        self.dataDir = dataDir
        self.train_path_info = train_path_info
        if completeDir is None:
            self.set_output_dir()
        else:
            self.completeDir=completeDir
            if not os.path.exists(completeDir):
                os.mkdir(completeDir)
            
        self.LOADED = False
        self.groups = []
        self.reuse_stats = False

        self.static_data_params = static_data_params
        self.network_params     = network_params

        params={}
        params['static_data_params']= static_data_params
        params['network_params']=network_params
        self.save_params(params,self.completeDir)
        
    def set_output_dir(self):
        dirName = input('directory Name pls: ')
        if not os.path.exists('networks'):
            os.mkdir('networks')
        
        completeDir = 'networks/{}'.format(dirName)
        if not os.path.exists('networks/{}'.format(dirName)):
            os.mkdir(completeDir)
        else:
            print('Path already exists.')
            A= input('Do you want to reuse the directory? Y/N: ')
            if A in ['Y','y','J','j']:
                pass
            else:
                dirName = input('New Directory Name pls: ')
                completeDir = 'networks/{}'.format(dirName)
                os.mkdir(completeDir)
        self.completeDir = completeDir

    def apply_stats(self,standardize,subjects_index):        
        if standardize=='look_at_time':
            assert len(self.data['X'][subjects_index])==len(self.trainMean)
            self.data['X'][subjects_index] = (self.data['X'][subjects_index]-self.trainMean)/self.trainstd
        elif standardize=='look_at_all_axis':
            output_shape = (len(subjects_index),1,1)
            trainMean = np.zeros(output_shape)+self.trainMean
            trainstd = np.zeros(output_shape)+self.trainstd
            self.data['X'][subjects_index] = (self.data['X'][subjects_index]-self.trainMean)/self.trainstd
            
    def standardizeData(self,standardize,subjects):
        """
        PROGRAM STRUCTURE IS UNCLEAR
        
        Works in place in order to save memory
        param: standardize one of {'look_at_time','look_at_trial',
            'look_at_car','look_at_all_axis',None}
            look_at_all_axis should be used for MEG data
        param subjects: subjects to which the standardize should be applied
        defined
        """
        if self.reuse_stats:
            self.apply_stats(standardize,subjects)
            return
        if 'look_at_time'==standardize:
            self.trainMean = np.mean(self.data['X'][subjects],axis=1,keepdims=True)# has shape: (subjects,1,channels)
            self.trainstd  = np.std(self.data['X'][subjects],axis=1,keepdims=True,ddof=1)
            reuse_stats = False # recompute stats also for test subjects
            
        elif 'look_at_car'==standardize:
            """for completeness, but not advised"""
            self.trainMean = np.mean(self.data['X'][subjects],axis=2,keepdims=True)# has shape: (subjects,samples,1)
            self.trainstd  = np.std(self.data['X'][subjects],axis=2,keepdims=True)
        
        elif 'look_at_trial'==standardize:
            raise ValueError
        
        elif 'look_at_all_axis'==standardize:
            # prefered mode.
            # can only be computed from training data
            output_shape = (len(subjects),1,1)
            self.trainMean = np.mean(self.data['X'][self.train_index])
            self.trainstd  = np.std(self.data['X'][self.train_index],ddof=1)
            print('Todo: Save trainMean and trainstd s.t. it can be used for test data')
            print('m: ',self.trainMean,' s: ',self.trainstd)
            self.reuse_stats = True
            
        elif standardize==None:
            return np.zeros(output_shape),np.ones(output_shape)
        
        elif standardize=='ema':
            print('Exponential moving average standardisation with alpha 5e-4')
            for subject in range(len(self.data['X'])):
                self.data['X'][subject] = utils.ema_substraction(self.data['X'][subject],alpha=5e-4)
            return
        elif standardize=='em_astd':
            alpha = 5e-4
            print('Exponential moving z score with alpha: {}'.format(alpha))
            for subject in range(len(self.data['X'])):
                self.data['X'][subject] = utils.em_avg_std(self.data['X'][subject],alpha=alpha)
            return
        else:
            print('Not standardizing data, choose one of {}'.format(['look_at_time','look_at_car',
                                                             'look_at_trial','look_at_all_axis']))  
            raise ValueError
        self.apply_stats(standardize,subjects)
        #condition_subjects_dict = {'dementia':[1],'control':[1],'mci':[1]}   
         
    def assembleLoad(self,condition_subjects_dict,**kwargs):
        """
        param: condition_subjects_dict e.g. {'mci':[1,2],'control':np.arange(10),'dementia':[1,2,3]}
        """
        
        loader = readSubjects.DataLoader(self.train_path_info,
            self.dataDir,self.static_data_params['channel_matches'],**kwargs)
        out = loader.make_Keras_data(condition_subjects_dict,fs=self.static_data_params['fs'],**kwargs)
        data = {}
        data['X'] = np.transpose(out[0],[0,2,1])
        data['Y'] = out[1]
        meta      = out[2]
        self.data_loaded = True
        return data,meta
        
    def startTrainFold(self,model,subjects_train_valid,
                       train_augment_pipeline,
                       care_memory=False,
                       valid_augment_pipeline=None,
                       callbacks_list=[]):
        
        #print('subjects for training: \n', subjects_train_valid)
        use_class_weights  = self.network_params['use_class_weights']
        fs                 = self.static_data_params['fs']
        standardize        = self.static_data_params['standardize']
        
        
        
        if use_class_weights:
            class_weights = class_weight.compute_class_weight('balanced',np.unique(self.data['Y']),self.data['Y'])
            class_weights = dict(enumerate(class_weights))
        else:
            class_weights = dict(enumerate([1,1]))
        print('class_weights: \n',class_weights)
        
        # GroupKFold is *not* necessary since each subject is considered a training sample
        # But StratifiedKFold is necessary to account for the imbalance by taking an equal percentage
        # of samples from each class
        gkf_valid = StratifiedKFold(n_splits=int(np.min([len(np.unique(subjects_train_valid))-1,10])),shuffle=True)
        
        print('num splits: ',gkf_valid.get_n_splits())
        self.train_index,self.valid_index = next(gkf_valid.split(np.ones((len(subjects_train_valid),1)),
            self.data['Y'], subjects_train_valid))
        
        subjects_valid = subjects_train_valid[self.valid_index]
        subjects_train = subjects_train_valid[self.train_index]
        #print('subjects_valid: \n ',subjects_valid,'subjects_train: \n ',subjects_train)
        # order of these 2 lines is important:
        # data from subjects train is used to calculate mean and std
        # if train is first set to mean=0,std=1 then we can not calculated mean,std for valid
        self.standardizeData(standardize,self.valid_index)#self.subjects_valid
        self.standardizeData(standardize,self.train_index)#self.subjects_train
        self.data['X'].flags.writeable = False
        self.data['Y'].flags.writeable = False
        
        if care_memory:
            raise NotImplementedError
        # Easy Implementation, consumes more memory but allows more flexibility
        else:
            print('Train Patients: \n',np.unique(subjects_train),
                '\n In Total: ',len(np.unique(subjects_train)))
            print('Valid Patients: \n',np.unique(subjects_valid),
                '\n In Total: ',len(np.unique(subjects_valid)))
            
            generator_train = GeneratorCNN.SlicingGeneratorAugment(self.static_data_params,self.network_params,train_augment_pipeline,debug=False)
            generator_valid = GeneratorCNN.SlicingGeneratorAugment(self.static_data_params,self.network_params,valid_augment_pipeline)
            
            generator_train.setData(self.data,self.train_index,self.meta)
            generator_valid.setData(self.data,self.valid_index,self.meta)
            
            callbacks_list.extend([callbacks.ModelCheckpoint(os.path.join(self.completeDir,'cp.ckpt'),
                                       monitor="val_mci_dem_F1",
                                       save_best_only=True,
                                       mode='max')])
            train_history = model.fit(generator_train,validation_data=generator_valid, 
                epochs=self.network_params['numTrainEpochs'],
                callbacks=callbacks_list,
                class_weight=class_weights,
                workers=self.network_params['workers'],
                use_multiprocessing=self.network_params['multiprocessing'],
                max_queue_size=20)
        model = keras.models.load_model(os.path.join(self.completeDir,'cp.ckpt'),compile=False)
        return model
    
    def get_optimizer(self):
        """Return optimizer specified by configuration."""
        optimizer         = self.network_params['optimizer']
        if type(optimizer)==str:
            optimizer = keras.optimizers.get(optimizer)    # Default parameters
            if 'learning_rate' in self.network_params.keys():
                lr     = self.network_params['learning_rate']
                if lr is not None:
                    optimizer = type(optimizer)(learning_rate=lr)
        return optimizer
    
    def data_params_description(self):
        """
        channel_matches: json with one fixed matching or 
                   numpy array with multiple matchings i.e. Pareto optimal 
        """
        pass
    
    def save_params(self,param_dict,params_dir):    
        with open(os.path.join(params_dir,'params.json'), 'w') as outfile:
            json.dump(param_dict, outfile,default=utils.serialize_np, indent=4, sort_keys=True)
    
    def load_params(self):
        with open(os.path.join(self.completeDir,'params.json'),'r') as f:
            params = json.load(f)
        self.static_data_params = params['static_data_params']
        self.network_params = params['network_params']
        for k in self.static_data_params.keys():
            if type(self.static_data_params[k])==list:
                self.static_data_params[k] = np.array(self.static_data_params[k])
        for k in self.network_params.keys():
            if type(self.network_params[k])==list:
                self.network_params[k] = np.array(self.network_params[k])
    
    def load_model(self,fold):
        model = keras.models.load_model(os.path.join(self.completeDir,'fold_{}'.format(fold),'model'),compile=False)
        return model    
        
    def prepare_fold(self,condition_subjects_dict_train, network, metrics):
        # save test and train subjects
        
        if self.network_params['num_classes'] is None:
            num_classes  = len(list(self.static_data_params.keys()))
            self.network_params['num_classes'] = num_classes
        optimizer = self.get_optimizer()
        readSubjects_params = self.static_data_params['readSubjects_params'] 
        
        #{'site_as_label':self.static_data_params['site_as_label']}
        self.data,self.meta = self.assembleLoad(condition_subjects_dict_train,**readSubjects_params)
        
        # time_steps,channels
        if 'channel_matches' not in self.static_data_params.keys() or self.static_data_params['channel_matches'] is None:
            num_channels = self.data['X'].shape[2]
        else:
            num_channels = self.static_data_params['channel_matches'].shape[-1]
        frame_samples = utils.time2samples(**{'fs':self.static_data_params['fs'],
            'time_container':   utils.Bunch(**self.network_params['frame_size'])})
        input_shape = (frame_samples,num_channels,1)
        print('Network has shape: ',input_shape)
        model = network(input_shape,network_params=self.network_params)
        model.summary()
        # Open the file
        with open(os.path.join(self.completeDir,'model_summary.py'),'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
        loss_str = 'categorical_crossentropy' if self.network_params['num_classes']>2 else 'binary_crossentropy'
        #import tensorflow as tf
        #run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
        model.compile(optimizer,loss=loss_str,metrics=metrics)#,options=run_opts)
        subjects_train_valid = np.array([str(sub['condition'])+str(sub['id']) for sub in self.meta['subjects']]).astype('str')
        return model, subjects_train_valid
    
    def start_main(self,network:keras.models.Model,
            train_augment_pipeline,
            valid_augment_pipeline=None,
            metrics=['acc'],
            callbacks_list=[]):
        """
        Interface       
        """
        if os.path.isfile(os.path.join(self.completeDir,'readme.txt')):
            print("Function 'start_main' was already run in this folder. \
                Either delete this folder or use 'continue_training'")
            return -1
        # For reprucibility:
        scriptName = os.path.basename(__file__)
        f=open(scriptName,'r')
        lines = f.readlines()
        with open(os.path.join(self.completeDir,scriptName),'w') as fcopy:
            for l in lines:
                fcopy.write(l)
        print('')

            #f.write('Used Data: 'self.dataDir+'\n')
        try:        
            with open(os.path.join(self.completeDir,'train_aug_pipe.pkl'), 'wb') as output:
                pickle.dump(train_augment_pipeline, output, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(self.completeDir,'valid_pipe.pkl'), 'wb') as output:
                pickle.dump(valid_augment_pipeline, output, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(self.completeDir,'architecture.pkl'), 'wb') as output:
                pickle.dump(network, output, pickle.HIGHEST_PROTOCOL)  
        except pickle.PicklingError:
            print('PicklingError')
            
        if 'n_folds' not in self.network_params.keys() or self.network_params['n_folds']==1:        
            #print('Todo: Better TrainTest split. If recording site is an important classification factor')
            #condition_subjects_dict_train,condition_subjects_dict_test = utils.split_subjects(
            #    self.static_data_params['subjects'],
            #    method=utils.split_ratio,**{'test_ratio':self.static_data_params['test_ratio']})    
            test_ratio = self.static_data_params['test_ratio']
            condition_subjects_dict_train,condition_subjects_dict_test = utils.split_wrt_site(
                self.static_data_params['subjects'],test_ratio)
            fold_dir = 'fold_'+str(0)    
            with open(os.path.join(self.completeDir,fold_dir,'test_subjects.json'),'w') as f:
                json.dump(condition_subjects_dict_test,f,default=utils.serialize_np)
            with open(os.path.join(self.completeDir,fold_dir,'train_subjects.json'),'w') as f:
                json.dump(condition_subjects_dict_train,f,default=utils.serialize_np)         
            n_folds = 1
            cv = [(condition_subjects_dict_train,condition_subjects_dict_test)]            
        else:        
            n_folds = self.network_params['n_folds']
            cv = utils.cv_split_wrt_site(self.static_data_params['subjects'],n_splits=n_folds)
            for k in range(n_folds):                
                fold_dir = 'fold_'+str(k)
                subdir = os.path.join(self.completeDir,fold_dir)
                if os.path.exists(subdir) and os.path.isdir(subdir):
                    shutil.rmtree(subdir)
                os.mkdir(subdir)           
                condition_subjects_dict_train,condition_subjects_dict_test = cv[k] 
                with open(os.path.join(self.completeDir,fold_dir,'test_subjects.json'),'w') as f:
                    json.dump(condition_subjects_dict_test,f,default=utils.serialize_np)
                with open(os.path.join(self.completeDir,fold_dir,'train_subjects.json'),'w') as f:
                    json.dump(condition_subjects_dict_train,f,default=utils.serialize_np)
                del condition_subjects_dict_train
        
        
        # If start_main runs up to this point function can not be called again.
        with open(os.path.join(self.completeDir,'readme.txt'),'w') as f:
            f.write('Used Data: '+self.dataDir+'\n')
            
        for k in range(n_folds):
            print('#########')
            fold_dir = 'fold_'+str(k)
            subdir = os.path.join(self.completeDir,fold_dir)
            
            with open(os.path.join(self.completeDir,fold_dir,'train_subjects.json'),'r') as f:
                condition_subjects_dict_train = json.load(f)
            model, subjects_train_valid = self.prepare_fold(condition_subjects_dict_train,network,metrics=metrics)
            model= self.startTrainFold(model,
                                       subjects_train_valid,
                                       train_augment_pipeline,
                                       care_memory=False,
                                       valid_augment_pipeline=valid_augment_pipeline,
                                       callbacks_list=callbacks_list)
            # Save Model
            model.save(os.path.join(self.completeDir,fold_dir,'model'))
            del self.data
        return model
    
    def check_model_exists(self,fold_dir):
        files_and_dirs = os.listdir(os.path.join(self.completeDir,fold_dir))
        if 'model' in files_and_dirs:
            cont = input('Model folder already exists. Do you want to continue? Y,N')
            if cont=='Y':
                return False
            else:
                raise ValueError
        else:
            return False
        
    def load_pickle(self,filename):
        if os.path.isfile(os.path.join(self.completeDir,filename)):
            with open(os.path.join(self.completeDir,filename), 'rb') as f:
                obj = pickle.load(f)
        else:
            obj = None
        return obj
        
    def continue_training(self,fold,network,metrics=['acc'],callbacks_list=[]):
        # check first that subjects all test folds do not overlap
        self.check_test_subjects_distinct()
        fold_dir = 'fold_'+str(fold)
        self.check_model_exists(fold_dir)
        
        train_augment_pipeline = self.load_pickle('train_aug_pipe.pkl')
        valid_augment_pipeline = self.load_pickle('valid_pipe.pkl')

        
        #with open(os.path.join(self.completeDir,'architecture.pkl'), 'rb') as f:
        #    network = pickle.load(f)
        with open(os.path.join(self.completeDir,fold_dir,'train_subjects.json'),'r') as f:
            condition_subjects_dict_train = json.load(f)
        

        self.load_params()
        model, subjects_train_valid = self.prepare_fold(condition_subjects_dict_train,
                                                        network,
                                                        metrics=metrics)
        model= self.startTrainFold(model,
                                   subjects_train_valid,
                                   train_augment_pipeline,
                                   care_memory=False,
                                   valid_augment_pipeline=valid_augment_pipeline,
                                   callbacks_list=callbacks_list)
        # Save Model
        model.save(os.path.join(self.completeDir,fold_dir,'model'))
        del self.data
        return model
        
    def test_model(self,model,test_pipe,fold,metrics=[],keep_data=False,data_loaded=False,
            subject_group="test_subjects.json",make_classification=True,iterative=False,
            feature_type="feature"):
        """
        1. Don't put too much effort in testing. Not every change must be tested with 5 Fold CV
        2. Make changes that give us more insight about the data, not just better results. 
        3. How shall we deal with the prediction for the multiple snapshots?
        4. How can we bring prediction and data into a visual comparison?
        5. Use enough senseful metrics. No metric cherry picking. Use Specifity, Sensitivity, Accuracy as everybody else.
        
        What we need to know in order to... 
            a) make a better classifier
                1. Influence of site.
                2. Influence of bad channels.
            b) explain classifier:
                1. Visual features
            c) sell classifier:
        """
        if "test" in subject_group:
            feature_type=feature_type+"_test"
        else:
            feature_type=feature_type+"_train"
        
        self.load_params()
        fold_dir = "fold_{}".format(fold)
        with open(os.path.join(self.completeDir,fold_dir,subject_group),'r') as f:
            condition_subjects_dict_test = json.load(f)
        #model,generator_test,readSubjects_params,condition_subjects_dict,iterative
        preds_soft, y_true, labels, all_ids, site = self._predict(model,test_pipe,
                                                    condition_subjects_dict_test,iterative,data_loaded=data_loaded)
        if make_classification:
            np.save(os.path.join(self.completeDir,r'fold_{}'.format(fold),'preds'),preds_soft)
            np.save(os.path.join(self.completeDir,r'fold_{}'.format(fold),'condition'),self.data['Y'])
            np.save(os.path.join(self.completeDir,r'fold_{}'.format(fold),'ids'),all_ids)     
            ids = ['{}{:03d}'.format(cond,nbr) for cond,nbr in list(zip(labels,all_ids))]
            df = pd.DataFrame({'y':np.argmax(y_true,axis=-1),'pred':np.argmax(np.mean(preds_soft,axis=1),axis=-1),'ids':ids,'site':site})
            df=df.set_index('ids')
            df.to_csv(os.path.join(self.completeDir,'fold_{}'.format(fold),'result.csv'))
            print(df.to_markdown())
        else:
            print('Feature output')
            np.save(os.path.join(self.completeDir,r'fold_{}'.format(fold),feature_type),preds_soft)
            np.save(os.path.join(self.completeDir,r'fold_{}'.format(fold),feature_type+'site'),site)
            np.save(os.path.join(self.completeDir,r'fold_{}'.format(fold),feature_type+'_y'), np.argmax(y_true,axis=-1))
            np.save(os.path.join(self.completeDir,r'fold_{}'.format(fold),feature_type+'_ids'), all_ids)
            
        if not keep_data and hasattr(self,'data'):
            del self.data
        return np.argmax(y_true,axis=-1),preds_soft,all_ids,labels
    
    def _predict(self, model,test_pipe,condition_subjects_dict,iterative,data_loaded):
        """
        handles the prediction with the network
        """
        readSubjects_params = self.static_data_params['readSubjects_params']
        generator_test = GeneratorCNN.SlicingGeneratorAugment(
                            self.static_data_params,self.network_params,
                            augmenter=test_pipe)
        output   = []
        labels   = []
        all_ids  = []
        site     = []        
        if iterative:
            splitted_subjects = utils.split_subject_dict(condition_subjects_dict)
            for sub_dicts in splitted_subjects:
                data,meta = self.assembleLoad(sub_dicts,**readSubjects_params)
                test_index = np.array([0]) 
                generator_test.setData(data,test_index,meta)
                # prediction has shape time_steps,classes
                prediction = model.predict(generator_test)
                output  += [prediction]
                labels  += [meta['subjects'][0]['condition']]
                all_ids += [meta['subjects'][0]['id']]
                site    += [meta['subjects'][0]['site']]
        else:
            if not data_loaded:
                self.data,self.meta = self.assembleLoad(condition_subjects_dict,**readSubjects_params)         
                self.standardizeData(self.static_data_params['standardize'],np.arange(len(self.data['Y'])))  
            subjects_test = np.array([str(sub['condition'])+str(sub['id']) for sub in self.meta['subjects']])
            for test_index in tqdm(np.arange(len(self.data['Y'])),position=0,leave=True):
                test_index = np.array([test_index]) 
                generator_test.setData(self.data,test_index,self.meta)
                # prediction has shape time_steps,classes
                prediction = model.predict(generator_test)
                output  += [prediction]
                labels  += [self.meta['subjects'][test_index[0]]['condition']]
                all_ids += [self.meta['subjects'][test_index[0]]['id']]
                site    += [self.meta['subjects'][test_index[0]]['site']]
            output=np.array(output)
            all_ids = np.array(all_ids)
            site = np.array(site)
            labels = np.array(labels)
            
        y_true_one_hot = np.zeros((len(labels),3))
        y_true = np.array([utils.condition_to_digit(lab) for lab in labels])
        y_true_one_hot[np.arange(len(labels)),y_true] = 1     
        return output, y_true_one_hot, labels, all_ids, site
            
    
    def evaluate3classes(self,ylabels,preds_soft,fold,metrics):    
        preds = np.argmax(np.mean(preds_soft,axis=1),axis=-1)
        from libs import calc_scores
        conf_mat = calc_scores.get_confusion_matrix(ylabels,preds)
        np.savetxt(os.path.join(self.completeDir,'fold_'+str(fold),'conf_mat.txt'),conf_mat.astype(int))
        
        gc.collect()
        # Aggregation of predictions:
        # 1. Average
        # Everything else is cumbersome. Before implementing more check how strong the
        # result changes if we leave out some snapshots.        
        
        
        # Visual Evaluation:
        # This is more challenging, but reveals more about what we can improve.
        # Focus on this evaluation.
        # Eventually this evaluation can give us information about how to combine the
        # predictions of each snapshot.
        # What:
        # a) Labels and signal
        # b) Model structure aka weights -> static independent of current input
        # c) Activation patterns -> dynamic direct connection to data input and label
        import matplotlib.pyplot as plt
        subjects_control  = np.where(ylabels==0)[0] 
        subjects_mci      = np.where(ylabels==1)[0] 
        subjects_dementia = np.where(ylabels==2)[0] 
        # label time series are assumed to be uncorrelated with the subject that means it
        # makes no sense to average_reduce them along the subject axis.
        # But it can make sense to calculate stats such as std for each time_axis.
        print('Todo: Wrap into method')
        """
        plt.figure()
        fig,axes = plt.subplots(1,3,figsize=(15,6))
        fig.subplots_adjust(top=0.8)
        axes[0].plot(preds_soft[subjects_dementia[0],:,0]);axes[0].set_title('Prob Class Control')
        axes[1].plot(preds_soft[subjects_dementia[0],:,1]);axes[1].set_title('Prob Class MCI')
        axes[2].plot(preds_soft[subjects_dementia[0],:,2]);axes[2].set_title('Prob Class Dementia') 
        fig.suptitle('Classificaton of a subject with dementia \n Softmax over time - \n (Describes not really the uncertainty)', y=0.98)
        plt.show()
        """
        # It might be interesting to compare classification results and signal
        # But most likely we can not see the relevant features by eye
        
        # Plain scalar metrics:
        # Cohen's Kappa: https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english
        # This is just for the paper.
        #visualisations.plot_confusion_matrix(ytestLarge.flatten(),pred_test.flatten())
        #conf_matrix_test  = calc_scores.get_confusion_matrix(ytestLarge,pred_test)

        #import tensorflow_addons as tfa
        #m = tfa.metrics.CohenKappa(num_classes=3, sparse_labels=True)
        #m.update_state(ylabels, np.array(preds))
        #print("Cohen's Kappa: ", m.result().numpy())
        y_true = np.zeros((len(ylabels),3))
        y_true[np.arange(len(ylabels)),ylabels]=1
        for m in metrics:
            m.update_state(y_true,np.mean(preds_soft,axis=1))
            print(m.name, ':', m.result())
        return conf_mat
        
        # save results for eventual offline analysis 
    def check_test_subjects_distinct(self):
        all_test_subjects=[]
        n_folds = [folddir for folddir in os.listdir(self.completeDir) if \
            os.path.isdir(os.path.join(self.completeDir,folddir)) and 'fold_' in folddir]
        n_folds = len(n_folds)
        for k in range(n_folds):                
            fold_dir = 'fold_'+str(k)
            subdir = os.path.join(self.completeDir,fold_dir)
            with open(os.path.join(self.completeDir,fold_dir,'test_subjects.json'),'r') as f:
                condition_subjects_dict_test = json.load(f)
            all_test_subjects+=[condition_subjects_dict_test]
        if not utils.suject_dicts_are_unique(all_test_subjects):
             raise ValueError('Check Test Subjects. Can not presume training when subject partitioning is corrupted.')

#if not 'readSubjects_params' in self.static_data_params.keys():
#    readSubjects_params = {'site_as_label':False}
#else:
#    readSubjects_params = self.static_data_params['readSubjects_params']
   
def evaluate(ylabels,preds_soft):
    """
    preds_soft: (subjects,time_steps,num_classes) 
    """
    num_classes = len(np.unique(ylabels))

    # label time series are assumed to be uncorrelated with the subject that means it
    # makes no sense to average_reduce them along the subject axis.
    # But it can make sense to calculate stats such as std for each time_axis.
    preds = np.argmax(np.mean(preds_soft,axis=1),axis=-1)
    from libs import calc_scores
    conf_mat = calc_scores.get_confusion_matrix(ylabels,preds)
    
    gc.collect()
    # Aggregation of predictions:
    # 1. Average
    # Everything else is cumbersome. Before implementing more check how strong the
    # result changes if we leave out some snapshots.        
    
    
    # Visual Evaluation:
    # This is more challenging, but reveals more about what we can improve.
    # Focus on this evaluation.
    # Eventually this evaluation can give us information about how to combine the
    # predictions of each snapshot.
    # What:
    # a) Labels and signal
    # b) Model structure aka weights -> static independent of current input
    # c) Activation patterns -> dynamic direct connection to data input and label
    import matplotlib.pyplot as plt
    
    subjects_control  = np.where(ylabels==0)[0] 
    subjects_mci      = np.where(ylabels==1)[0] 
    subjects_dementia = np.where(ylabels==2)[0] 
   
    print('Todo: Wrap into method')
    plt.figure()
    fig,axes = plt.subplots(1,3,figsize=(15,6))
    axes[0].plot(preds_soft[subjects_dementia[0],:,0]);axes[0].set_title('Prob Class Control')
    axes[1].plot(preds_soft[subjects_dementia[0],:,1]);axes[1].set_title('Prob Class MCI')
    axes[2].plot(preds_soft[subjects_dementia[0],:,2]);axes[2].set_title('Prob Class Dementia') 
    fig.suptitle('Classificaton of a subject with dementia \n Softmax over time - \n (Describes not really the uncertainty)')
    plt.show()
    
    # It might be interesting to compare classification results and signal
    # But most likely we can not see the relevant features by eye
    
    # Plain scalar metrics:
    # Cohen's Kappa: https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english
    # This is just for the paper.
    visualisations.plot_confusion_matrix(ylabels,
        preds,class_labels=['control','mci','dementia'],fontsize=14,
        title='')#'Predicted Condition for Evaluation Set'

    import tensorflow_addons as tfa
    m = tfa.metrics.CohenKappa(num_classes=num_classes, sparse_labels=True)
    m.update_state(ylabels, np.array(preds))
    print("Cohen's Kappa: ", m.result().numpy())
    return conf_mat

# bugged lw not definied
def make_roc_curve_in_sklearn_style(ylabels,preds_soft):
    
    from sklearn.preprocessing import label_binarize
    from itertools import cycle
    from sklearn import metrics 
    
    y_test = label_binarize(ylabels, classes=[0, 1, 2])
    y_score = np.mean(preds_soft,axis=1)
    n_classes = y_score.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
      fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
      roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])


   

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
      mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
          label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
          color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
          label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
          color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
      plt.plot(fpr[i], tpr[i], color=color, lw=lw,
              label='ROC curve of class {0} (area = {1:0.2f})'
              ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

  
   
from tensorflow.keras import layers,models
def get_Frequenzgang_of_temporal_layer(model,fs,layer_name,channel_name):
    """
    model = models.Model(full_model.layers[0].input,full_model.get_layer(layer_name).output)
    """
    channel_idx = channelname2idx(channel_name)
    print(channel_idx)
    t = np.arange(0,1,1/fs)
    amplitude64 = []
    for k in np.arange(64):
        s = np.repeat(1*np.sin(np.pi*2*k*t).reshape(128,1),64,axis=1)
        filtered = model.predict(np.expand_dims(s,axis=0))
        amplitude64 += [abs(np.fft.fft(np.squeeze(filtered[0,channel_idx]).T)[:,k])]
    amplitude64 = np.array(amplitude64)
    fig = plt.figure(figsize=(15,15))
    for x in np.arange(8):
        for y in np.arange(8):
            ax=fig.add_subplot(8,8,x*8+y+1)
            ax.plot(np.arange(64),amplitude64.reshape(8,8,64)[y,x])
    plt.show()
            
#get_Frequenzgang(m,128,'conv2d','Cz')
            
def get_temporal_transfer_function(model,layer_name,channel_name):
    channel_idx = channelname2idx(channel_name)
    time_frame = model.layers[0].input_shape[0][1]
    num_channels = model.layers[0].input_shape[0][2]
    data = np.random.randn(1,time_frame,num_channels)
    
    m2 = models.Model(model.layers[0].input,model.get_layer('temporal_conv').output)
    filtered = m2.predict(data)
    plt.figure()
    plt.subplot(121)
    plt.plot(abs(np.fft.fft(data))[:int(time_frame//2),channel_idx])
    plt.title('Amplitude')
    plt.subplot(122)
    plt.plot(abs(np.fft.fft(filtered[0,channel_idx].T))[:int(time_frame//2)])
    plt.xlabel('f in [Hz]')
    plt.title('')
    plt.show()


