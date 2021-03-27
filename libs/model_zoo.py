#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Models for BioMag Competition function for the BioMag Dementia Challenge.

(c) 2021 Marc Steffen Seibel, Technische Universitaet Ilmenau
'''

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.backend import int_shape
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, BatchNormalization, GlobalMaxPooling2D, Activation, Cropping2D, Dropout

def circular_dilated_conv():
    return


class PairwiseChannelConv(tf.keras.layers.Layer):
    def __init__(self,
            filters, kernel_size,aggregation,
            dilation_rates=None,activation=None, use_bias=True,padding='SAME',
            *args,**kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation_rates = np.array(dilation_rates)
        self.activation = activation
        self.use_bias = use_bias
        self.strides = 1
        self.aggregation = aggregation
        
        super().__init__(*args, **kwargs)
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(
                self.filters,
                self.kernel_size,
                padding=self.padding,
                use_bias=self.use_bias)
        self.conv.build(input_shape)
        self.act = layers.Activation(self.activation)

    def call(self, inputs, **kwargs):
        meg_channel_axis = 1
        
        num_channels = K.int_shape(inputs)[meg_channel_axis]
        X_split = tf.split(inputs,num_channels,meg_channel_axis)
        print('Building layer. This may take some time.')
        max_pairwise_similarity = []
        for i,ch_i in enumerate(X_split):
            # I hope that pairwise_sim_ij is cleared whenever the next i,ch_i iteration starts
            # but eventually tensorflow will produce recreate always a new tensor without burrying 
            # the old ones.
            pairwise_sim_ij = [tf.nn.conv2d(ch_i+ch_j,
                        self.conv.weights[0], 
                        padding=self.padding,
                        strides=self.strides) for j,ch_j in enumerate(X_split) if i!=j] 
            
            pairwise_sim_ij = tf.concat(pairwise_sim_ij,axis=1)
            if self.aggregation=='max':
                max_pairwise_similarity+=[tf.reduce_max(pairwise_sim_ij,axis=1)]
            elif self.aggregation == 'mean':
                max_pairwise_similarity+=[tf.reduce_mean(pairwise_sim_ij,axis=1)]
            else:
                raise ValueError
            
        return tf.stack(max_pairwise_similarity,axis=1)



def inception_model(input_shape,network_params):
    
    num_classes = network_params['num_classes']
    num_channels = input_shape[1]
    x = Input(shape=input_shape, name="eeg_input")

    # Temporal Inception 1

    short_term_1 = Conv2D(filters=24, kernel_size=(1, 1), use_bias=False, padding="same")(x)
    short_term_1 = BatchNormalization()(short_term_1)
    short_term_1 = Activation('relu')(short_term_1)

    short_term_2 = Conv2D(filters=16, kernel_size=(3, 1), use_bias=False, padding="same")(x)
    short_term_2 = BatchNormalization()(short_term_2)
    short_term_2 = Activation('relu')(short_term_2)

    short_term_3 = Conv2D(filters=16, kernel_size=(5, 1), use_bias=False, padding="same")(x)
    short_term_3 = BatchNormalization()(short_term_3)
    short_term_3 = Activation('relu')(short_term_3)

    short_term_4 = MaxPooling2D(pool_size=(3, 1), strides=(1, 1), padding='same')(x)
    short_term_4 = Conv2D(filters=16, kernel_size=(1, 1), use_bias=False, padding="same")(short_term_4)
    short_term_4 = BatchNormalization()(short_term_4)
    short_term_4 = Activation('relu')(short_term_4)

    eeg_after_short = layers.concatenate([short_term_1, short_term_2, short_term_3, short_term_4])
    eeg_after_short = MaxPooling2D(pool_size=(3, 1), strides=(2, 1))(eeg_after_short)

    # Spatio-Temporal Inception 2

    mid_term_1 = Conv2D(filters=32, kernel_size=(1, num_channels), use_bias=False)(eeg_after_short)
    mid_term_1 = Cropping2D(cropping=((2, 2), (0, 0)), data_format=None)(mid_term_1)
    mid_term_1 = BatchNormalization()(mid_term_1)
    mid_term_1 = Activation('relu')(mid_term_1)

    mid_term_2 = Conv2D(filters=24, kernel_size=(3, num_channels), use_bias=False)(eeg_after_short)
    mid_term_2 = Cropping2D(cropping=((1, 1), (0, 0)), data_format=None)(mid_term_2)
    mid_term_2 = BatchNormalization()(mid_term_2)
    mid_term_2 = Activation('relu')(mid_term_2)

    mid_term_3 = Conv2D(filters=24, kernel_size=(5, num_channels), use_bias=False)(eeg_after_short)
    mid_term_3 = BatchNormalization()(mid_term_3)
    mid_term_3 = Activation('relu')(mid_term_3)

    mid_term_4 = MaxPooling2D(pool_size=(3, 1), strides=(1, 1))(eeg_after_short)
    mid_term_4 = Conv2D(filters=24, kernel_size=(1, num_channels), use_bias=False)(mid_term_4)
    mid_term_4 = Cropping2D(cropping=((1, 1), (0, 0)), data_format=None)(mid_term_4)
    mid_term_4 = BatchNormalization()(mid_term_4)
    mid_term_4 = Activation('relu')(mid_term_4)

    eeg_after_mid = layers.concatenate([mid_term_1, mid_term_2, mid_term_3, mid_term_4])
    eeg_after_mid = Dropout(0.25, noise_shape=(1, 1, 104))(eeg_after_mid)

    # Spatio-Temporal Inception 3

    long_term_1 = Conv2D(filters=64, kernel_size=(1, 1), use_bias=False, padding="same")(eeg_after_mid)
    long_term_1 = BatchNormalization()(long_term_1)
    long_term_1 = Activation('relu')(long_term_1)

    long_term_2 = Conv2D(filters=48, kernel_size=(3, 1), use_bias=False, padding="same")(eeg_after_mid)
    long_term_2 = BatchNormalization()(long_term_2)
    long_term_2 = Activation('relu')(long_term_2)

    long_term_3 = Conv2D(filters=48, kernel_size=(5, 1), use_bias=False, padding="same")(eeg_after_mid)
    long_term_3 = BatchNormalization()(long_term_3)
    long_term_3 = Activation('relu')(long_term_3)

    long_term_4 = MaxPooling2D(pool_size=(3, 1), strides=(1, 1), padding='same')(eeg_after_mid)
    long_term_4 = Conv2D(filters=48, kernel_size=(1, 1), use_bias=False, padding="same")(long_term_4)
    long_term_4 = BatchNormalization()(long_term_4)
    long_term_4 = Activation('relu')(long_term_4)

    eeg_after_long = layers.concatenate([long_term_1, long_term_2, long_term_3, long_term_4])
    eeg_after_long = MaxPooling2D(pool_size=(7, 1), strides=(5, 1))(eeg_after_long)
    eeg_after_long = Dropout(0.25, noise_shape=(1, 1, int(eeg_after_long.shape[3])))(eeg_after_long)

    main = GlobalAveragePooling2D()(eeg_after_long)

    final = Dropout(0.4)(main)

    
    if num_classes==2:
        y = layers.Dense(1,activation='sigmoid')(final)
    else:
        y = layers.Dense(num_classes,activation='softmax',)(final) # kernel_constraint=max_norm(.5)
    
    
    model = models.Model(x, y)

    return model
    
"""
# off diag
X_split = tf.split(tf.random.normal((2,256,160,16)),160,axis=2)
t1 = time.time()
num_tests=2
for _ in range(num_tests):
    aggregation = 'max'
    pws = []
    for k in range(160)[:-1]:
        right = X_split[slice(k+1,160,1)]+X_split[k]
        left  = X_split[slice(0,k+1,1)]+X_split[k]
        tmp = tf.concat([right,left],axis=0)
        if aggregation=='max':
            # source_axis has only 1 element, therefore this can be seen as squeeze
            pws+=[tf.reduce_max(tmp,axis=(0))]
print((time.time()-t1)/num_tests)

t1 = time.time()
for _ in range(num_tests):
    aggregation = 'max'
    pws = []
    for k in range(160):
        tmp = X_split[slice(k+1,160,1)]+X_split[k]
        if aggregation=='max':
            pws+=[tmp]#tf.reduce_max(tmp,axis=0)
print((time.time()-t1)/num_tests)
num_tests = 100
source_axis=2
t1 = time.time()
for _ in range(num_tests):
    pws2 = []
    for i,ch_i in enumerate(X_split):
        # I hope that pairwise_sim_ij is cleared whenever the next i,ch_i iteration starts
        # but eventually tensorflow will produce recreate always a new tensor without burrying 
        # the old ones.
        pairwise_sim_ij = [ch_i+ch_j for j,ch_j in enumerate(X_split) if j>i]
        if len(pairwise_sim_ij):
            pairwise_sim_ij = tf.concat(pairwise_sim_ij,axis=source_axis)
        else:
            pairwise_sim_ij=[]
        pws2+=[pairwise_sim_ij]
print((time.time()-t1)/num_tests)
"""    
    
class PairwiseChannelAdd(tf.keras.layers.Layer):
    def __init__(self,aggregation,
            mode='off-diag',
            source_axis=1,
            *args,**kwargs,
            ):
        """
        params: mode: 'half','off-diag','full', 
        different modes make sense for aggregation==None
        """
        self.aggregation = aggregation
        self.mode = mode
        self.source_axis = source_axis
            
        super().__init__(*args, **kwargs)
        
    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        source_axis = self.source_axis
        print(source_axis)
        num_channels = K.int_shape(inputs)[source_axis]
        # consider tf.unstack
        X_split = tf.split(inputs,num_channels,source_axis)
        
        #print(X_split)
        
        #max_pairwise_similarity = tf.TensorArray(tf.float32, num_channels-1)

        #print(max_pairwise_similarity.shape)
        #return tf.concat(max_pairwise_similarity,axis=source_axis)
        return self._pairwise_add(X_split,num_channels,source_axis,self.mode,self.aggregation)
    
    @tf.function
    def _pairwise_add(self,X_split,num_channels,source_axis,mode,aggregation):
        # slow on start especially with backpro for first time
        # https://github.com/tensorflow/tensorflow/issues/40517
        print('Tracing')
        #tf.print('executing')
        max_pairwise_similarity = tf.TensorArray(dtype=tf.float32, size=num_channels-1)#[]
        for k in range(num_channels)[:-1]:                
            #print('before concat: ',pairwise_sim_ij)
            # throws error for mode=='half' must add if len(~)!=0
            if mode=='half':
                pairwise_sim_ij = X_split[slice(k+1,160,1)]+X_split[k]
            elif mode=='off-diag':
                right = X_split[slice(k+1,160,1)] + X_split[k]
                left  = X_split[slice(0,k+1,1)]   + X_split[k]
                pairwise_sim_ij = tf.concat([right,left],axis=0)
            else:
                raise ValueError
            #print('after: ',pairwise_sim_ij)
            if aggregation=='max':
                connec =tf.reduce_max(pairwise_sim_ij,axis=(0,-2))
            elif aggregation == 'mean':
                connec = tf.reduce_mean(pairwise_sim_ij,axis=(0,-2))
            elif aggregation==None:
                connec = pairwise_sim_ij
            else:
                raise ValueError
            max_pairwise_similarity = max_pairwise_similarity.write(k, connec)
            #max_pairwise_similarity+=[connec]
        max_pairswise_similarity = max_pairwise_similarity.stack()     
        return tf.transpose(max_pairswise_similarity,[1,2,0,3])
    
    #https://keras.io/guides/serialization_and_saving/#custom-objects
    def get_config(self):
        return {"aggregation": self.aggregation,
               "mode":self.mode,
               "source_axis":self.source_axis}

    # There's actually no need to define `from_config` here, since returning
    # `cls(**config)` is the default behavior.
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
#@tf.function    
def _pairwise_add_depr(X_split,num_channels,source_axis,mode,aggregation):
    # slow with for even if structure is constant
    # https://github.com/tensorflow/tensorflow/issues/40517
    print('Tracing')
    tf.print('executing')
    max_pairwise_similarity = []
    for k in range(num_channels)[:-1]:                
        #print('before concat: ',pairwise_sim_ij)
        # throws error for mode=='half' must add if len(~)!=0
        if mode=='half':
            pairwise_sim_ij = X_split[slice(k+1,160,1)]+X_split[k]
        elif mode=='off-diag':
            right = X_split[slice(k+1,160,1)] + X_split[k]
            left  = X_split[slice(0,k+1,1)]   + X_split[k]
            pairwise_sim_ij = tf.concat([right,left],axis=0)
        else:
            raise ValueError
        #print('after: ',pairwise_sim_ij)
        if aggregation=='max':
            connec =tf.reduce_max(pairwise_sim_ij,axis=0)
        elif aggregation == 'mean':
            connec = tf.reduce_mean(pairwise_sim_ij,axis=0)
        elif aggregation==None:
            connec = pairwise_sim_ij
        else:
            raise ValueError
        #max_pairwise_similarity.write(k, connec)
        max_pairwise_similarity+=[connec]
    return tf.concat(max_pairwise_similarity,axis=source_axis)


"""

X_split = tf.split(tf.random.normal((2,256,160,16)),160,axis=2)
out = _pairwise_add(X_split,160,2,'off-diag','mean')

from libs import model_zoo
reload(model_zoo)
m = model_zoo.ConnectivityModel((512,160,1),network_params)
m.compile('adam','categorical_crossentropy')
m.fit(tf.random.normal((2,512,160,1)),tf.constant([[0,0,1],[0,1,0]]),epochs=100)

out = m.predict(tf.random.normal((2,512,160,1)))
loss = tf.losses.categorical_crossentropy(tf.constant([[0,0,1],[0,1,0]]),out)

"""



class CircularPadding():
    def __init__(self,padsize):
        self.padsize
        pass
    def build(self):
         self.permute = tf.eye(self.input_shape[1])
    def compute_output_shape(self):
         return (self.input_shape[0]+2*padsize)
    def call(self,x,**kwargs):
         return self.permute@x

# idea of implementation by https://github.com/tensorflow/tensorflow/issues/37278
class SharedDilatedConv(tf.keras.layers.Layer):
    def __init__(self,
            filters,kernel_size,
            dilation_rates=None,
            activation=None,
            use_bias=True,
            *args,
            **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = 'VALID'
        self.dilation_rates = np.array(dilation_rates)
        self.activation = activation
        self.use_bias = use_bias
        self.strides = 1
        
        super().__init__(*args, **kwargs)
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(
                self.filters,
                self.kernel_size,
                padding=self.padding,
                dilation_rate=self.dilation_rates[0],
                use_bias=self.use_bias)
        self.conv.build()
        self.act = layers.Activation(self.activation)

    def call(self, inputs, **kwargs):
        x1 = self.conv(inputs)
       
        x = [tf.nn.conv2d(
        inputs,
        self.conv.weights[0], 
        padding=self.padding,
        strides=self.strides, 
        dilations=dilation
        ) for dilation in self.dilation_rates[1:]]
        x = tf.concat([x1]+x,axis=1)
        
        if self.use_bias:
            x = x + self.conv.weights[1]
        x = self.act(x)
        return x
        
def ConnectivityModel(input_shape,network_params):
    """
    input_shape: (time,channels,1)
    """
    use_bn = network_params['use_bn']
    do_ratio = network_params['do_ratio']
    num_classes = network_params['num_classes']
    if 'connectivity_mode' in network_params.keys():
        connectivity_mode = network_params['connectivity_mode']
    else:
        connectivity_mode = "off-diag"
    if 'aggregation' in network_params.keys():
        aggregation = network_params['aggregation']
    else:
        aggregation = 'mean'
    # input_shape: (time_steps,N_SPHARA_harmonics)
    numChannels = input_shape[1]
    inp = layers.Input(input_shape)
    
    c1 = layers.Conv2D(16,kernel_size=(35,1),padding='same')(inp)
    c1 = layers.AveragePooling2D(pool_size=(2,1),strides=(2,1),padding='same')(c1)
    c1 = layers.Activation('elu')(c1)
    c1 = PairwiseChannelAdd(aggregation=aggregation,mode=connectivity_mode,source_axis=2)(c1)
    numPairs = K.int_shape(c1)[-2]
    c1 = layers.Conv2D(16,kernel_size=(1,numPairs),padding='valid',use_bias=~use_bn,name='spatial_conv')(c1)    
    if use_bn:
        c1 = layers.BatchNormalization()(c1)    
    # square
    c1 = layers.Activation(K.square)(c1)
    # (1,35 |7)
    c1 = layers.AveragePooling2D(pool_size=(25,1),strides=(7,1))(c1)
    # not existent
    #c1 = layers.Conv2D(128,kernel_size=(1,13),padding='same', kernel_constraint=max_norm(2.),use_bias=True)(c1)
    c1 = layers.Activation(K.log)(c1)
    
    c1 = layers.Flatten()(c1)
    c1 = layers.Dropout(do_ratio)(c1)
    
    if num_classes==2:
        c1 = layers.Dense(1,activation='sigmoid')(c1)
    else:
        c1 = layers.Dense(num_classes,activation='softmax',)(c1) # kernel_constraint=max_norm(.5)
        
    
    return models.Model(inp,c1) 
    
         
def channelProcessing(x,numFilters,downsample=False):
    convTime1 = layers.Conv2D(numFilters,(1,5),padding='same')(x)
    #convTime1 = layers.BatchNormalization()(convTime1)
    convTime1 = layers.Activation('relu')(convTime1)
    if downsample:
        convTime1 = layers.MaxPooling2D(pool_size=(1,2),padding='same',strides=(1,2))(convTime1)
        convTime1 = layers.Conv2D(numFilters,(1,5),padding='same')(convTime1)
    else:
        convTime1 = layers.Conv2D(numFilters,(1,5),padding='same')(convTime1)
    #convTime1 = layers.BatchNormalization()(convTime1)
    act = layers.Activation('relu')(convTime1)
    return act

def interChannelProcessing(x1,x2,numFilters):
    x = layers.concatenate([x1,x2],axis=1)
    convTime1 = layers.Conv2D(numFilters,(2,5),padding='same')(x)
    #convTime1 = layers.BatchNormalization()(convTime1)
    convTime1 = layers.Activation('relu')(convTime1)
    convTime1 = layers.Conv2D(numFilters,(2,5),padding='same')(convTime1)
    #convTime1 = layers.BatchNormalization()(convTime1)
    act = layers.Activation('relu')(convTime1)
    return act
    
def mergeProcessing(x,numFilters):
    convTime1 = layers.Conv2D(numFilters,(1,5),padding='same')(x)
    #convTime1 = layers.BatchNormalization()(convTime1)
    convTime1 = layers.Activation('relu')(convTime1)
    #net3MaxPoolDownNoBatchNormMSE
    convTime1 = layers.MaxPooling2D(pool_size=(1,2),padding='same',strides=(1,2))(convTime1)
    convTime1 = layers.Conv2D(numFilters,(1,5),padding='same')(convTime1)
    #convTime1 = layers.BatchNormalization()(convTime1)
    convTime1 = layers.Activation('relu')(convTime1)
    
    convTime1 = layers.Conv2D(numFilters,(4,9),padding='same')(x)
    #convTime1 = layers.BatchNormalization()(convTime1)
    convTime1 = layers.Activation('relu')(convTime1)
    convTime1 = layers.Conv2D(numFilters,(7,1),padding='valid')(convTime1)
    #convTime1 = layers.BatchNormalization()(convTime1)
    act = layers.Activation('relu')(convTime1)
    return act

def resBlock(x,numFilters,use_bn=False,axis=2):
    """
    axis=2 applies Convolution for time axis
    axis=1 applies convolution for sensor channel (SPAHARA_harmonics) axis
    """
    assert axis in (1,2)
    if axis==2:
        kernel = (1,5)
    else:
        kernel = (5,1)
    convTime1 = layers.Conv2D(numFilters,kernel,padding='same')(x)
    if use_bn:
        convTime1 = layers.BatchNormalization()(convTime1)
    convTime1 = layers.Activation('relu')(convTime1)
    convTime1 = layers.Conv2D(numFilters,kernel,padding='same')(convTime1)
    if use_bn:
        convTime1 = layers.BatchNormalization()(convTime1)
    convTime1 = layers.Activation('relu')(convTime1)
    if numFilters!=int_shape(x)[-1]:
        x = layers.Conv2D(numFilters,kernel_size=(1,1))(x)
    return layers.Add()([x,convTime1])

def resBlock2D(x,numFilters):
    convTime1 = layers.Conv2D(numFilters,(3,3),padding='same')(x)
    convTime1 = layers.BatchNormalization()(convTime1)
    convTime1 = layers.Activation('relu')(convTime1)
    convTime1 = layers.Conv2D(numFilters,(3,3),padding='same')(convTime1)
    convTime1 = layers.BatchNormalization()(convTime1)
    convTime1 = layers.Activation('relu')(convTime1)
    if numFilters != int_shape(x)[-1]:
        x = layers.Conv2D(numFilters, kernel_size=(1, 1))(x)
    return layers.Add()([x,convTime1])

def conv_block(x, stage, branch, nb_filter, dropout_rate=None,weight_decay=1e-4, use_conv2d=False):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    #print('f',nb_filter)
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    if use_conv2d:
        inter_channel = nb_filter * 4
    else:
        inter_channel = nb_filter * 2
    x = layers.BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    #x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = layers.Activation('relu', name=relu_name_base+'_x1')(x)
    x = layers.Conv2D(inter_channel, kernel_size=(1, 1),padding='same', name=conv_name_base+'_x1', use_bias=False)(x)

    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = layers.BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    #x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = layers.Activation('relu', name=relu_name_base+'_x2')(x)
    #x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    if use_conv2d:
        x = layers.Conv2D(nb_filter, kernel_size=(3, 3), name=conv_name_base+'_x2', use_bias=False,padding='same')(x)
    else:
        x = layers.Conv2D(nb_filter, kernel_size=(1, 3), name=conv_name_base+'_x2', use_bias=False,padding='same')(x)

    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)
    return x
    
    
def denseBlock(x,numFilters, stacks=4):
    convTime1 = layers.Conv2D(numFilters,(1,3),padding='same')(x)
    #convTime1 = layers.BatchNormalization()(convTime1)
    prev = layers.Activation('relu')(convTime1)
    for k in range(stacks-1):        
        conv = layers.Conv2D(numFilters,(1,3),padding='same')(prev)
        #convTime1 = layers.BatchNormalization()(convTime1)
        conv  = layers.Activation('relu')(conv)
        prev = layers.Concatenate(axis=-1)([conv,prev])
        #prev += [layers.Conv2D(numFilters//stacks,kernel_size=(1,1))(conc)]
    out = layers.Conv2D(numFilters//stacks,kernel_size=(1,1))(prev)
    return out

def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4,use_conv2d=False, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''
    #print('f',nb_filter,'gr',growth_rate)
    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, nb_filter, dropout_rate, weight_decay,use_conv2d=use_conv2d)
        concat_feat = layers.concatenate([concat_feat, x],axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate
        #print(nb_filter)
    return concat_feat, nb_filter

def makeDENSEencoder(input_shape=(None,None)):
    inp = layers.Input(shape=input_shape)
    # preprocessing with same filters on all emg channels, but without connecting information 
    # between emg channels
    convTime1 = layers.Conv2D(16,(1,9),padding='same')(inp)
    #convTime1 = layers.BatchNormalization()(convTime1)
    convTime1 = layers.Activation('relu')(convTime1)
    convTime1 = layers.MaxPooling2D(pool_size=(1,2),padding='same',strides=(1,2))(convTime1)
    denseBlock0 = denseBlock(convTime1,24,stacks=4)
    
    denseBlock1 = denseBlock(denseBlock0,24,stacks=4)
    down1 = layers.MaxPooling2D(pool_size=(1,2),padding='same',strides=(1,2))(denseBlock1)
    
    denseBlock2 = denseBlock(down1,32,stacks=4)
    down2 = layers.MaxPooling2D(pool_size=(1,2),padding='same',strides=(1,2))(denseBlock2)
    
    denseBlock3 = denseBlock(down2,32,stacks=4)
    denseBlock3 = denseBlock(denseBlock3,32,stacks=4)
    down3 = layers.MaxPooling2D(pool_size=(1,2),padding='same',strides=(1,2))(denseBlock3)
    
    denseBlock4 = denseBlock(down3,32,stacks=4)
    denseBlock4 = denseBlock(denseBlock4,32,stacks=4)
    return models.Model(inp,denseBlock4)

def makeDENSEdecoder(x):
    isModel=True
    if isModel:
        xin = x.output
    #enc = layers.Input(input=x)
    up1  = layers.Conv2DTranspose(128,kernel_size=(1,3),strides=(1,2),padding='same')(xin)
    #up1  = layers.BatchNormalization()(up1)
    up1   = layers.Activation('relu')(up1)
    
    up1 = denseBlock(up1,128,stacks=8)
    
    up1  = layers.Conv2DTranspose(64,kernel_size=(1,3),strides=(1,2),padding='same')(up1)
    #up1  = layers.BatchNormalization()(up1)
    up1   = layers.Activation('relu')(up1)
    up1 = denseBlock(up1,64,stacks=12)
    
    up1  = layers.Conv2DTranspose(32,kernel_size=(1,3),strides=(1,2),padding='same')(up1)
    #up1  = layers.BatchNormalization()(up1)
    up1   = layers.Activation('relu')(up1)
    up1 = denseBlock(up1,32,stacks=8)
    
    up1  = layers.Conv2DTranspose(16,kernel_size=(1,3),strides=(1,2),padding='same')(up1)
    #up1  = layers.BatchNormalization()(up1)
    up1   = layers.Activation('relu')(up1)
    up1 = denseBlock(up1,32,stacks=4)
    
    
    #up1  = layers.BatchNormalization(axis=1)(up1)
    out = layers.Conv2D(1,kernel_size=(1,1),activation='sigmoid')(up1)
    
    m = models.Model(x.input,out)
    return m

    
def makeAECencoder(input_shape=(None,None)):
    inp = layers.Input(shape=input_shape)
    # preprocessing with same filters on all emg channels, but without connecting information
    # between emg channels
    convTime1 = layers.Conv2D(8,(1,5),padding='same')(inp)
    #convTime1 = layers.BatchNormalization()(convTime1)
    convTime1 = layers.Activation('relu')(convTime1)
    convTime1 = layers.MaxPooling2D(pool_size=(1,2),padding='same',strides=(1,2))(convTime1)
    convTime1  = layers.Conv2D(16,kernel_size=(1,5),padding='same')(convTime1)
    #convTime1 = layers.BatchNormalization()(convTime1)
    convTime1 = layers.Activation('relu')(convTime1)
    
    convTime1 = resBlock(convTime1,16)
    
    ch0 = layers.Lambda(lambda x : x[:,0,:])(convTime1)
    ch1 = layers.Lambda(lambda x : x[:,1,:])(convTime1)
    ch2 = layers.Lambda(lambda x : x[:,2,:])(convTime1)
    ch3 = layers.Lambda(lambda x : x[:,3,:])(convTime1)
    
    ch0 = layers.Reshape((1,int_shape(ch0)[1], int_shape(ch0)[2]))(ch0)
    ch1 = layers.Reshape((1,int_shape(ch1)[1], int_shape(ch1)[2]))(ch1)
    ch2 = layers.Reshape((1,int_shape(ch2)[1], int_shape(ch2)[2]))(ch2)
    ch3 = layers.Reshape((1,int_shape(ch3)[1], int_shape(ch3)[2]))(ch3)
    
    ch0_res = channelProcessing(ch0,16,True)
    ch1_res = channelProcessing(ch1,16,True)
    ch2_res = channelProcessing(ch2,16,True)
    ch3_res = channelProcessing(ch3,16,True)
        
    
    ch0 = channelProcessing(ch0_res,16,False)
    ch1 = channelProcessing(ch1_res,16,False)
    ch2 = channelProcessing(ch2_res,16,False)
    ch3 = channelProcessing(ch3_res,16,False)
    
    ch0 = layers.Add()([ch0,ch0_res])
    ch1 = layers.Add()([ch1,ch1_res])
    ch2 = layers.Add()([ch2,ch2_res])
    ch3 = layers.Add()([ch3,ch3_res])
    
    """
    ch01 = interChannelProcessing(ch0,ch1,16)
    ch02 = interChannelProcessing(ch0,ch2,16)
    ch03 = interChannelProcessing(ch0,ch3,16)
    ch12 = interChannelProcessing(ch1,ch2,16)
    ch13 = interChannelProcessing(ch1,ch3,16)
    ch23 = interChannelProcessing(ch2,ch3,16)
    
    ch0code = layers.concatenate([ch0,ch01,ch02,ch03],axis=1)
    ch1code = layers.concatenate([ch1,ch01,ch12,ch13],axis=1)
    ch2code = layers.concatenate([ch2,ch02,ch12,ch23],axis=1)
    ch3code = layers.concatenate([ch3,ch03,ch13,ch23],axis=1)
    
    ch0 = mergeProcessing(ch0code,32)
    ch1 = mergeProcessing(ch1code,32)
    ch2 = mergeProcessing(ch2code,32)
    ch3 = mergeProcessing(ch3code,32)
    """
    ch0_res = channelProcessing(ch0,numFilters=32,downsample=True)
    ch1_res = channelProcessing(ch1,numFilters=32,downsample=True)
    ch2_res = channelProcessing(ch2,numFilters=32,downsample=True)
    ch3_res = channelProcessing(ch3,numFilters=32,downsample=True)
    
    ch0 = channelProcessing(ch0_res,32,False)
    ch1 = channelProcessing(ch1_res,32,False)
    ch2 = channelProcessing(ch2_res,32,False)
    ch3 = channelProcessing(ch3_res,32,False)
    
    ch0 = layers.Add()([ch0,ch0_res])
    ch1 = layers.Add()([ch1,ch1_res])
    ch2 = layers.Add()([ch2,ch2_res])
    ch3 = layers.Add()([ch3,ch3_res])
    
    code = layers.concatenate([ch0,ch1,ch2,ch3],axis=1)
    code = resBlock(code,32)
    
    m = models.Model(inp,code)
    return m


def makeAECdecoder(x):
    isModel=True
    if isModel:
        xin = x.output
    #enc = layers.Input(input=x)
    up1  = layers.Conv2DTranspose(32,kernel_size=(1,5),strides=(1,2),padding='same')(xin)
    #up1  = layers.BatchNormalization()(up1)
    up1   = layers.Activation('relu')(up1)
    
    up1 = resBlock(up1,32)
    
    up1  = layers.Conv2DTranspose(16,kernel_size=(1,5),strides=(1,2),padding='same')(up1)
    #up1  = layers.BatchNormalization()(up1)
    up1   = layers.Activation('relu')(up1)
    up1 = resBlock(up1,16)
    
    
    up1  = layers.Conv2DTranspose(16,kernel_size=(1,5),strides=(1,2),padding='same')(up1)
    #up1  = layers.BatchNormalization()(up1)
    up1   = layers.Activation('relu')(up1)
    up1 = resBlock(up1,16)
    #up1  = layers.BatchNormalization(axis=1)(up1)
    out = layers.Conv2D(1,kernel_size=(1,1),activation='sigmoid')(up1)
    
    m = models.Model(x.input,out)
    return m
    
def transition_block1D(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution,  optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 

    x = layers.BatchNormalization(name=conv_name_base+'_bn')(x)
    #x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = layers.Activation('relu', name=relu_name_base)(x)
    x = layers.Conv2D(int(nb_filter * compression), kernel_size=(1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.MaxPooling2D((1, 2), strides=(1, 2), name=pool_name_base)(x)
    return x

def transition_block2D(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution,  optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = layers.BatchNormalization(name=conv_name_base+'_bn')(x)
    #x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = layers.Activation('relu', name=relu_name_base)(x)
    x = layers.Conv2D(int(nb_filter * compression), kernel_size=(1, 1), name=conv_name_base, use_bias=False)(x)
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x

def makeLSTMencoder(input_shape, growth_rate=4, nb_filter=64,return_sequences=True):
    # input: (None,4,TimeSamples,1)
    global concat_axis
    concat_axis =-1
    inp = layers.Input(input_shape)
    # little hack to use one lstm for each of the 4 channels
    # sharing one LSTM for each channel -> drawback: Channels arent
    # features anymore
    conv1 = layers.Conv2D(32,kernel_size=(1,3),padding='same')(inp)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.MaxPooling2D(pool_size=(1,2),strides=(1,2))(conv1)
    #dense_block takes: (x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True)
    
    denseBlock0, nb_filter = dense_block(conv1,0,4,nb_filter,growth_rate)
    denseBlock0 = transition_block1D(denseBlock0,0,nb_filter)
    
    denseBlock1, nb_filter = dense_block(denseBlock0,1,12,nb_filter,growth_rate)
    denseBlock1 = transition_block1D(denseBlock1,1,nb_filter)
    
    denseBlock2, nb_filter = dense_block(denseBlock1,2,10,nb_filter,growth_rate)
    denseBlock2 = transition_block1D(denseBlock2,2,nb_filter)

    #denseBlock3, nb_filter = dense_block(denseBlock2,3,16,nb_filter,growth_rate)
    #denseBlock3 = transition_block1D(denseBlock3,3,nb_filter)
    # returns:(None,4,prevTimesteps, LSTMunits)
    lstm1 = layers.TimeDistributed(layers.LSTM(128,return_sequences=return_sequences))(denseBlock2)
    
    return models.Model(inp,lstm1)


def makeLSTMencoderSTFT(input_shape, growth_rate=4, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,return_sequences=True):
    # input: (None,4,TimeSamples,1)
    global concat_axis
    use_conv2d = True
    concat_axis =-1
    inp = layers.Input(input_shape)
    # little hack to use one lstm for each of the 4 channels
    # sharing one LSTM for each channel -> drawback: Channels arent
    # features anymore
    conv1 = layers.Conv2D(32,kernel_size=(3,3),padding='same')(inp)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    # reducing time, keeping frequency
    conv1 = layers.MaxPooling2D(pool_size=(1,2),strides=(1,2))(conv1)
    #dense_block takes: (x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True)
    
    denseBlock0, nb_filter = dense_block(conv1,0,4,nb_filter,growth_rate,use_conv2d=use_conv2d)
    denseBlock0 = transition_block2D(denseBlock0,0,nb_filter)
    
    denseBlock1, nb_filter = dense_block(denseBlock0,1,12,nb_filter,growth_rate,use_conv2d=use_conv2d)
    denseBlock1 = transition_block2D(denseBlock1,1,nb_filter)
    
    denseBlock2, nb_filter = dense_block(denseBlock1,2,10,nb_filter,growth_rate,use_conv2d=use_conv2d)
    denseBlock2 = transition_block2D(denseBlock2,2,nb_filter)

    #denseBlock3, nb_filter = dense_block(denseBlock2,3,16,nb_filter,growth_rate)
    #denseBlock3 = transition_block1D(denseBlock3,3,nb_filter)
    # returns:(None,4,prevTimesteps, LSTMunits)
    lstm1 = layers.TimeDistributed(layers.LSTM(128,return_sequences=return_sequences))(denseBlock2)
    
    return models.Model(inp,lstm1)

def tinyClassifierSTFT(input_shape):
    inp = layers.Input(input_shape)
    c1 = layers.Conv2D(32,kernel_size=(3,3),padding='same')(inp)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(c1)
    c1 = resBlock2D(c1,64)
    c1 = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(c1)
    c1 = resBlock2D(c1,128)
    c1 = layers.AveragePooling2D(pool_size=(2,2),strides=(2,2))(c1)
    c1 = layers.Flatten()(c1)
    c1 = layers.Dense(4,activation='relu',use_bias=True)(c1)
    c1 = layers.Dense(1,activation='sigmoid',use_bias=True)(c1)
    return models.Model(inp,c1)



def smallestclfSTFT(input_shape):
    numChannels = input_shape[-1]
    inp = layers.Input(input_shape)
    c1 = layers.Lambda(lambda x: K.expand_dims(x,1))(inp)
    c1 = layers.Permute([4,2,3,1])(c1)
    # new shape: n_Sensor_channels,Time,Frequency,Features 
    c1 = layers.Conv3D(16,kernel_size=(1,3,3),padding='same',
                       #kernel_regularizer=regularizers.l2(0.01)
                       )(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation(activation='relu')(c1)
    c1 = layers.Conv3D(16, kernel_size=(1, 3, 3), activation='relu', padding='same',
                       #kernel_regularizer=regularizers.l2(0.01)
                       )(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation(activation='relu')(c1)
    c1 = layers.MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2))(c1)

    c1 = layers.Conv3D(32, kernel_size=(1, 3, 3), padding='same',
                       #kernel_regularizer=regularizers.l2(0.01)
                       )(c1)
    #c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation(activation='relu')(c1)
    c1 = layers.Conv3D(32, kernel_size=(1, 3, 3), activation='relu', padding='same',
                       #kernel_regularizer=regularizers.l2(0.01)
                       )(c1)
    #c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation(activation='relu')(c1)
    c1 = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(c1)
    singleGAP2D = []
    for chnum in range(numChannels):
        ch0 = layers.Lambda(lambda x : x[:,chnum])(c1)
        singleGAP2D += [layers.GlobalAveragePooling2D()(ch0)]
    c1 = layers.concatenate(singleGAP2D)
    #c1 = layers.Flatten()(c1)
    #c1 = layers.Dropout(0.1)(c1)
    #c1 = layers.Dense(8, activation='relu')(c1)
    c1 = layers.Dense(1,activation='sigmoid',
                      #kernel_regularizer=regularizers.l2(0.01)
                      )(c1)
    return models.Model(inp,c1)

def SpatialNetOffline(input_shape,network_params):
    """Not converging at all"""
    # input_shape: 512,10,11    
    inp = layers.Input(input_shape)
    # 512,10,11,1
    c1 = layers.Lambda(lambda x: K.expand_dims(x,-1))(inp)
    # new shape (10,11,time_steps,1) First dim: from Nasion to Inion, 2nd From Left to Right Ear
    c1 = layers.Permute([2,3,1,4])(c1)
    
    # This Spatial Filtering can be regarded as similar to Laplacian Referencing 
    # Padding is valid, because it is assumed that the outer positions are not relevant for motor task
    # Further lots of empty channels in the outer area
    c1 = layers.Conv3D(32,kernel_size=(1,1,64),padding='valid')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('elu')(c1)
    c1 = layers.Conv3D(32,kernel_size=(1,1,64),padding='valid')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('elu')(c1)
    c1 = layers.MaxPooling3D((1,1,2),(1,1,2))(c1)
    
    c1 = layers.Conv3D(32,kernel_size=(7,7,1),padding='valid')(c1)
    
    c1 = layers.Conv3D(64,kernel_size=(1,1,32),activation='relu',padding='valid')(c1)
    c1 = layers.Conv3D(32,kernel_size=(3,3,1),padding='valid')(c1)
    c1 = layers.Conv3D(32,kernel_size=(1,1,32),activation='relu',padding='valid')(c1)
    #c1 = layers.MaxPooling3D((2,2,1),(2,2,1))(c1)
    c1 = layers.Conv3D(128,kernel_size=(1,1,32),activation='relu',padding='valid')(c1)
    c1 = layers.MaxPooling3D((1,1,2),(1,1,2))(c1)
    # Dimension Reduction with Bottleneck
    c1 = layers.Conv3D(32,kernel_size=(1,1,1),activation='relu',padding='valid')(c1)
    
    c1 = layers.Flatten()(c1)
    c1 = layers.Dense(1,'sigmoid')(c1)
    return models.Model(inp,c1,name='SpatialNetOffline')



def SpatialNet(input_shape,network_params):
    """Not converging at all"""        
    inp = layers.Input(input_shape)
    c1 = layers.Lambda(lambda x: K.expand_dims(x,1))(inp)
    # new shape: N_SPHARA_harmonics, time_steps, features
    c1 = layers.Permute([3,2,1])(c1)
    # new shape (10,11,time_steps,1) First dim: from Nasion to Inion, 2nd From Left to Right Ear
    c1 = layers.Lambda(lambda x: tf.gather(x,electrodes2D,batch_dims=0,axis=1))(c1)
    
    # This Spatial Filtering can be regarded as similar to Laplacian Referencing 
    # Padding is valid, because it is assumed that the outer positions are not relevant for motor task
    # Further lots of empty channels in the outer area
    c1 = layers.Conv3D(32,kernel_size=(1,1,64),padding='valid')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('elu')(c1)
    c1 = layers.Conv3D(32,kernel_size=(6,6,1),padding='valid')(c1)
    
    c1 = layers.Conv3D(64,kernel_size=(1,1,32),activation='relu',padding='valid')(c1)
    c1 = layers.Conv3D(32,kernel_size=(3,3,1),padding='valid')(c1)
    c1 = layers.Conv3D(32,kernel_size=(1,1,32),activation='relu',padding='valid')(c1)
    c1 = layers.MaxPooling3D((2,2,1),(2,2,1))(c1)
    c1 = layers.Conv3D(128,kernel_size=(1,1,32),activation='relu',padding='valid')(c1)
    c1 = layers.MaxPooling3D((1,1,2),(1,1,2))(c1)
    # Dimension Reduction with Bottleneck
    c1 = layers.Conv3D(32,kernel_size=(1,1,1),activation='relu',padding='valid')(c1)
    
    c1 = layers.Flatten()(c1)
    c1 = layers.Dense(1,'sigmoid')(c1)
    return models.Model(inp,c1,name='SpatialNet')


def SPHARA1Dclf(input_shape,use_bn=False,do_ratio=.0):
    # input_shape: (time_steps,N_SPHARA_harmonics)
    numChannels = input_shape[-1]
    inp = layers.Input(input_shape)
    c1 = layers.Lambda(lambda x: K.expand_dims(x,1))(inp)
    c1 = layers.Permute([3,2,1])(c1)
    # new shape: N_SPHARA_harmonics, time_steps, features
    c1 = layers.Dropout(do_ratio,noise_shape=[None,numChannels,1,1])(c1)
    #c1 = layers.BatchNormalization(axis=1)(c1)
    # resBlock(x,numFilters,use_bn=False,axis=2)
    c1  = resBlock(c1,16,use_bn=use_bn,axis=2)
    c1  = layers.MaxPooling2D(pool_size=(1,2),padding='same',strides=(1,2))(c1)

    c1  = resBlock(c1,32,use_bn=use_bn,axis=2)
    c1  = layers.MaxPooling2D(pool_size=(1,2),padding='same',strides=(1,2))(c1)
    
    c1  = resBlock(c1,32,use_bn=use_bn,axis=2)
    c1  = layers.MaxPooling2D(pool_size=(1,2),padding='same',strides=(1,2))(c1)
    
    c1  = resBlock(c1,64,use_bn=use_bn,axis=1)
    c1  = layers.MaxPooling2D(pool_size=(2,1),padding='same',strides=(2,1))(c1)
    
    c1  = resBlock(c1,64,use_bn=use_bn,axis=1)
    c1  = layers.MaxPooling2D(pool_size=(2,1),padding='same',strides=(2,1))(c1)
    numFeatures = K.int_shape(c1)[3]    
    #c1 = layers.Dropout(do_ratio,noise_shape=[None,1,1,numFeatures])(c1)
    
    numChannels = K.int_shape(c1)[1]
    print('numChannels',numChannels)
    """
    singleGAP2D = []
    for chnum in range(numChannels):
        ch0 = layers.Lambda(lambda x : x[:,chnum])(c1)
        singleGAP2D += [layers.GlobalAveragePooling1D()(ch0)
    c1 = layers.concatenate(singleGAP2D)
    """
    
    #c1 = layers.Flatten()(c1)
    #c1 = layers.Dropout(0.1)(c1)
    #c1 = layers.Dense(8, activation='relu')(c1)
    c1 = layers.Flatten()(c1)
    c1 = layers.Dense(1,activation='sigmoid',
                      #kernel_regularizer=regularizers.l2(0.01)
                      )(c1)
    return models.Model(inp,c1) 

def SPHARA1DclfFirstChannels(input_shape,use_bn=False,do_ratio=.0):
    # input_shape: (time_steps,N_SPHARA_harmonics)
    numChannels = input_shape[-1]
    inp = layers.Input(input_shape)
    c1 = layers.Lambda(lambda x: K.expand_dims(x,1))(inp)
    c1 = layers.Permute([3,2,1])(c1)
    # new shape: N_SPHARA_harmonics, time_steps, features
    c1 = layers.Dropout(do_ratio,noise_shape=[None,numChannels,1,1])(c1)
    #c1 = layers.BatchNormalization(axis=1)(c1)
    # resBlock(x,numFilters,use_bn=False,axis=2)
    c1  = resBlock(c1,16,use_bn=use_bn,axis=1)
    c1  = layers.MaxPooling2D(pool_size=(2,1),padding='same',strides=(2,1))(c1)

    c1  = resBlock(c1,32,use_bn=use_bn,axis=1)
    c1  = layers.MaxPooling2D(pool_size=(2,1),padding='same',strides=(2,1))(c1)
    
    c1  = resBlock(c1,32,use_bn=use_bn,axis=2)
    c1  = layers.MaxPooling2D(pool_size=(1,2),padding='same',strides=(1,2))(c1)
    
    c1  = resBlock(c1,64,use_bn=use_bn,axis=2)
    c1  = layers.MaxPooling2D(pool_size=(1,2),padding='same',strides=(1,2))(c1)
    
    c1  = resBlock(c1,64,use_bn=use_bn,axis=2)
    c1  = layers.MaxPooling2D(pool_size=(1,2),padding='same',strides=(1,2))(c1)
    numFeatures = K.int_shape(c1)[3]    
    #c1 = layers.Dropout(do_ratio,noise_shape=[None,1,1,numFeatures])(c1)
    
    numChannels = K.int_shape(c1)[1]
    print('numChannels',numChannels)
    """
    singleGAP2D = []
    for chnum in range(numChannels):
        ch0 = layers.Lambda(lambda x : x[:,chnum])(c1)
        singleGAP2D += [layers.GlobalAveragePooling1D()(ch0)
    c1 = layers.concatenate(singleGAP2D)
    """
    
    #c1 = layers.Flatten()(c1)
    #c1 = layers.Dropout(0.1)(c1)
    #c1 = layers.Dense(8, activation='relu')(c1)
    c1 = layers.Flatten()(c1)
    c1 = layers.Dense(1,activation='sigmoid',
                      #kernel_regularizer=regularizers.l2(0.01)
                      )(c1)
    return models.Model(inp,c1) 

from tensorflow.keras.constraints import max_norm
def ShallowConvNet_Multiplication_to_Addition(input_shape,network_params):
    """
    Riedmueller 2015 overfitted
    Neural networks are bad for fitting y=x1*x2. 
    But artificats can like respiration are multiplicatively modulated with eeg.
    We can transform y: y_new = log(y)=log(x1)+log(x2)
    """
    use_bn=network_params['use_bn']
    do_ratio=network_params['do_ratio']
    # input_shape: (time_steps,N_SPHARA_harmonics)
    numChannels = input_shape[-1]
    inp = layers.Input(input_shape)
    c1 = layers.Lambda(lambda x: K.expand_dims(x,1))(inp)
    c1 = layers.Permute([3,2,1])(c1)
    c1_log = layers.Activation(K.abs)(c1)
    c1_log = layers.Activation(K.log)(c1_log)
    c1_sig = layers.Activation(K.sign)(c1)
    c1_log = layers.Multiply()([c1_sig,c1_log])
    # new shape: N_SPHARA_harmonics, time_steps, features
    numChannels = K.int_shape(c1)[1]
    
    c1 = layers.Conv2D(48,kernel_size=(1,13),padding='same', kernel_constraint=max_norm(2.),use_bias=True)(c1)
    
    c1_log = layers.Conv2D(48,kernel_size=(1,13),padding='same', kernel_constraint=max_norm(2.),use_bias=True)(c1_log)
    
    c1 = layers.concatenate([c1,c1_log],axis=-1)
    c1 = layers.Conv2D(64,kernel_size=(numChannels,1),padding='valid', kernel_constraint=max_norm(2.),use_bias=True)(c1)    
    if use_bn:
        c1 = layers.BatchNormalization()(c1)    
    # square
    c1 = layers.Activation(K.square)(c1)
    # (1,35 |7)
    c1 = layers.AveragePooling2D(pool_size=(1,35),strides=(1,7))(c1)
    # not existent
    #c1 = layers.Conv2D(128,kernel_size=(1,13),padding='same', kernel_constraint=max_norm(2.),use_bias=True)(c1)
    c1 = layers.Activation(K.log)(c1)
    
    print('numChannels',numChannels)
    
    c1 = layers.Flatten()(c1)
    c1 = layers.Dropout(do_ratio)(c1)
    c1 = layers.Dense(1,activation='sigmoid', kernel_constraint=max_norm(.5))(c1)
    return models.Model(inp,c1) 


def SpatialTemporalMultiClass(input_shape,network_params,**kwargs):
    """
    input_shape: (time,channels,1)
    """
    use_bn=network_params['use_bn']
    do_ratio=network_params['do_ratio']
    num_classes = network_params['num_classes']
    
    inp = layers.Input(input_shape)
    numChannels = K.int_shape(inp)[2]
    print('numChannels',numChannels)
    
    c1 = layers.Conv2D(16,kernel_size=(3,numChannels),padding='valid',use_bias=True,name='temporal_conv')(inp)
    c1 = layers.Activation('elu')(c1)
    if use_bn:
        c1 = layers.BatchNormalization()(c1)
        

        
    #####
    ##### First Block
    #####
    c11 = layers.Conv2D(16,kernel_size=(35,1),strides=(1,1),padding='same',use_bias=~use_bn,name='c11stride')(c1)
    c11 = layers.Activation('relu')(c11)   
    if use_bn:
        c11 = layers.BatchNormalization()(c11)
    c11 = layers.MaxPooling2D(pool_size=(2,1),strides=(2,1),padding='same')(c11)
    c11 = layers.Conv2D(32,kernel_size=(5,1),strides=(1,1),padding='same',use_bias=~use_bn,name='c11conv')(c11)
    c11 = layers.Activation('relu')(c11) 
    if use_bn:
        c11 = layers.BatchNormalization()(c11)
        
    #####
    ##### SECOND BLOCK
    #####
    c12 = layers.Conv2D(32,kernel_size=(5,1),strides=(1,1),padding='same',use_bias=~use_bn)(c11)
    c12 = layers.Activation('relu')(c12)
    if use_bn:
        c12 = layers.BatchNormalization()(c12)
    c12 = layers.MaxPooling2D(pool_size=(2,1),strides=(2,1),padding='same')(c12)
    
    c1 = layers.Conv2D(32,kernel_size=(5,1),strides=(1,1),padding='same',use_bias=~use_bn)(c12)
    c1 = layers.Activation('relu')(c1)
    if use_bn:
        c11 = layers.BatchNormalization()(c11)
    
    c1 = layers.GlobalAveragePooling2D()(c1)
    if do_ratio>0:
        c1 = layers.Dropout(do_ratio)(c1)
    if num_classes==2:
        c1 = layers.Dense(1,activation='sigmoid')(c1)
    else:
        c1 = layers.Dense(num_classes,activation='softmax',)(c1) # kernel_constraint=max_norm(.5)
    return models.Model(inp,c1)

def SpatialAverageModel(input_shape,network_params):
    use_bn=network_params['use_bn']
    do_ratio=network_params['do_ratio']
    num_classes = network_params['num_classes']
    
    inp = layers.Input(input_shape)
    # bias not needed when using BatchNorm
    c1 = layers.Conv2D(32,kernel_size=(5,1),strides=(2,1),padding='valid',use_bias=~use_bn,name='temporal_conv_relu')(inp)
    if use_bn:
        c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(32,kernel_size=(13,1),padding='valid',use_bias=True,name='temporal_conv')(c1)
    # if multiple channels are provided take the mean s.t. the amount of extracted spatial information
    # is minimized as opossed to using spatial filters
    c1 = layers.Lambda(lambda x: K.mean(x,axis=2,keepdims=True))(c1)
    c1 = layers.Activation(K.square)(c1)
    # (1,35 |7)
    c1 = layers.AveragePooling2D(pool_size=(25,1),strides=(7,1))(c1)
    # not existent
    #c1 = layers.Conv2D(128,kernel_size=(1,13),padding='same', kernel_constraint=max_norm(2.),use_bias=True)(c1)
    c1 = layers.Activation(K.log)(c1)
    c1 = layers.Flatten()(c1)
    c1 = layers.Dropout(do_ratio)(c1)
    if num_classes==2:
        c1 = layers.Dense(1,activation='sigmoid')(c1)
    else:
        c1 = layers.Dense(num_classes,activation='softmax',)(c1) # kernel_constraint=max_norm(.5)
    return models.Model(inp,c1) 



def Channel_Decision_Small_Shared_Out(input_shape,network_params):
    if not 'use_bn' in network_params.keys():
        use_bn= True
    else:
        use_bn = network_params['use_bn']
    if not 'do_ratio' in network_params.keys():
        do_ratio= 0.5
    else:
        do_ratio = network_params['do_ratio']
    if not 'num_classes' in network_params.keys():
        num_classes = 3
    else:
        num_classes = network_params['num_classes']
    
    inp = layers.Input(input_shape)
    numChannels = K.int_shape(inp)[2]
    # bias not needed when using BatchNorm
    c1 = layers.Conv2D(32,kernel_size=(13,1),strides=(1,1),padding='same',use_bias=False,name='temporal_conv_relu')(inp)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.AveragePooling2D(pool_size=(2,1),strides=(2,1))(c1)
    fused = False
    if use_bn:
        c1 = layers.BatchNormalization(fused=fused)(c1)
        
    c1 = layers.Conv2D(32,kernel_size=(5,1),padding='same',use_bias=False,name='temporal_conv2')(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.AveragePooling2D(pool_size=(2,1),strides=(2,1))(c1)
    if use_bn:
        c1 = layers.BatchNormalization(fused=fused)(c1)
    
    c1 = layers.Conv2D(32,kernel_size=(5,1),padding='same',use_bias=False,name='temporal_conv3')(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.AveragePooling2D(pool_size=(2,1),strides=(2,1))(c1)
    if use_bn:
        c1 = layers.BatchNormalization(fused=fused)(c1)
    
    
    print(c1)
    # create layer that is reused 
    if num_classes==2:
        clf_out = layers.Dense(1,activation='sigmoid',use_bias=False)
    else:
        clf_out = layers.Dense(num_classes,activation='softmax',use_bias=False)
    
    classifiers  =[]
    # these layers have no params, but sharing them makes a better (I guess, since less ops)
    # computational graph (and summary)
    
    #flat_it = layers.Flatten()
    #drop_it = layers.Dropout(do_ratio)
    
    for i in range(numChannels):
    # Slicing the ith channel:
        _c1 = layers.Lambda(lambda x: x[:, :, i])(c1)
        _c1 = layers.Flatten()(_c1)
        if do_ratio!=0:
            _c1 = layers.Dropout(do_ratio)(_c1)
        _c1 = clf_out(_c1)
        classifiers+=[_c1]
    # bundle all predictions and form one output.
    classifiers = layers.Average()(classifiers)
    return models.Model(inp,classifiers) 

def Channel_Decision_Small_Single_Out(input_shape,network_params):
    use_bn=network_params['use_bn']
    do_ratio=network_params['do_ratio']
    num_classes = network_params['num_classes']
    
    inp = layers.Input(input_shape)
    numChannels = K.int_shape(inp)[2]
    # bias not needed when using BatchNorm
    c1 = layers.Conv2D(32,kernel_size=(5,1),strides=(2,1),padding='same',use_bias=False,name='temporal_conv_relu')(inp)
    c1 = layers.Activation('relu')(c1)
    if use_bn:
        c1 = layers.BatchNormalization()(c1)
    
    c1 = layers.Conv2D(32,kernel_size=(13,1),padding='same',use_bias=False,name='temporal_conv2')(c1)
    # if multiple channels are provided take the mean s.t. the amount of extracted spatial information
    # is minimized as opossed to using spatial filters
    c1 = layers.Activation('relu')(c1)
    # (1,35 |7)
    c1 = layers.AveragePooling2D(pool_size=(25,1),strides=(7,1))(c1)
    c1 = layers.Activation('relu')(c1)
    classifiers =[]
    for i in range(numChannels):
    # Slicing the ith channel:
        out = layers.Lambda(lambda x: x[:, :, i])(c1)
        out = layers.Flatten()(out)
        if do_ratio!=0:
            out = layers.Dropout(do_ratio)(out)
        if num_classes==2:
            out = layers.Dense(1,activation='sigmoid',use_bias=False)(out)
        else:
            out = layers.Dense(num_classes,activation='softmax',use_bias=False)(out) # kernel_constraint=max_norm(.5)
        classifiers+=[out]
    
    # bundle all predictions and form one output.
    classifiers  = layers.concatenate(classifiers,axis=1)
    classifiers = layers.Lambda(lambda x: K.mean(x,axis=1))(classifiers)
    return models.Model(inp,classifiers) 


def Channel_Decision_Small(input_shape,network_params):
    use_bn=network_params['use_bn']
    do_ratio=network_params['do_ratio']
    num_classes = network_params['num_classes']
    
    inp = layers.Input(input_shape)
    numChannels = K.int_shape(inp)[2]
    # bias not needed when using BatchNorm
    c1 = layers.Conv2D(32,kernel_size=(5,1),strides=(2,1),padding='same',use_bias=False,name='temporal_conv_relu')(inp)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(32,kernel_size=(13,1),padding='same',use_bias=False,name='temporal_conv2')(c1)
    # if multiple channels are provided take the mean s.t. the amount of extracted spatial information
    # is minimized as opossed to using spatial filters
    c1 = layers.Activation(K.square)(c1)
    # (1,35 |7)
    c1 = layers.AveragePooling2D(pool_size=(25,1),strides=(7,1))(c1)
    c1 = layers.Activation(K.log)(c1)
    classifiers =[]
    for i in range(numChannels):
    # Slicing the ith channel:
        out = layers.Lambda(lambda x: x[:, :, i])(c1)
        out = layers.Flatten()(out)
        if do_ratio!=0:
            out = layers.Dropout(do_ratio)(out)
        if num_classes==2:
            out = layers.Dense(1,activation='sigmoid',use_bias=False)(out)
        else:
            out = layers.Dense(num_classes,activation='softmax',use_bias=False)(out) # kernel_constraint=max_norm(.5)
        classifiers+=[out]
    return models.Model(inp,classifiers) 



def Channel_Decision_Model(input_shape,network_params):
    use_bn=network_params['use_bn']
    do_ratio=network_params['do_ratio']
    num_classes = network_params['num_classes']
    
    inp = layers.Input(input_shape)
    numChannels = K.int_shape(inp)[2]
    # bias not needed when using BatchNorm
    c1 = layers.Conv2D(16,kernel_size=(3,1),strides=(2,1),padding='same',use_bias=False,name='temporal_conv_relu')(inp)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(16,kernel_size=(13,1),padding='same',use_bias=False,name='temporal_conv2')(c1)
    c1 = layers.Activation('relu')(c1)

    c1 = layers.Conv2D(32,kernel_size=(3,1),strides=(2,1),padding='same',use_bias=False,name='temporal_conv_relu2')(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(32,kernel_size=(13,1),padding='same',use_bias=False,name='temporal_conv3')(c1)
    c1 = layers.Activation('relu')(c1)
    
    
    c1 = layers.Conv2D(32,kernel_size=(13,1),strides=(2,1),
        padding='same',use_bias=False,name='temporal_conv4')(c1)
    c1 = layers.Activation('relu')(c1)
    
    # if multiple channels are provided take the mean s.t. the amount of extracted spatial information
    # is minimized as opossed to using spatial filters
    #c1 = layers.Activation(K.square)(c1)
    # (1,35 |7)
    #c1 = layers.AveragePooling2D(pool_size=(25,1),strides=(7,1))(c1)
    #c1 = layers.Activation(K.log)(c1)
    classifiers =[]
    for i in range(numChannels):
    # Slicing the ith channel:
        out = layers.Lambda(lambda x: x[:, :, i])(c1)
        out = layers.Flatten()(out)
        if do_ratio!=0:
            out = layers.Dropout(do_ratio)(out)
        if num_classes==2:
            out = layers.Dense(1,activation='sigmoid',use_bias=False)(out)
        else:
            out = layers.Dense(num_classes,activation='softmax',use_bias=False)(out) # kernel_constraint=max_norm(.5)
        classifiers+=[out]
    return models.Model(inp,classifiers) 

    
def ShallowConvNetMultiClassLinear(input_shape,network_params):
    """
    input_shape: (time,channels,1)
    Riedmueller 2015 overfitted
    """
    use_bn=network_params['use_bn']
    do_ratio=network_params['do_ratio']
    num_classes = network_params['num_classes']
    
    inp = layers.Input(input_shape)
    numChannels = K.int_shape(inp)[2]
    print('numChannels',numChannels)
    
    # 
    c1 = layers.Conv2D(16,kernel_size=(5,1),strides=(2,1),padding='valid',use_bias=False,name='temporal_conv_relu')(inp)
    #if use_bn:
    #    c1 = layers.BatchNormalization()(c1)
    #c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(16,kernel_size=(13,1),padding='same',use_bias=False,name='temporal_conv')(c1)
    c1 = layers.Conv2D(16,kernel_size=(1,numChannels),padding='valid',use_bias=False,name='spatial_conv')(c1)    
    # square
    #c1 = layers.Activation(K.square)(c1)
    # (1,35 |7)
    c1 = layers.AveragePooling2D(pool_size=(25,1),strides=(7,1))(c1)
    # not existent
    #c1 = layers.Conv2D(128,kernel_size=(1,13),padding='same', kernel_constraint=max_norm(2.),use_bias=True)(c1)
    #c1 = layers.Activation(K.log)(c1)
    
    c1 = layers.Flatten()(c1)
    c1 = layers.Dropout(do_ratio)(c1)
    if num_classes==2:
        c1 = layers.Dense(1,activation='sigmoid')(c1)
    else:
        c1 = layers.Dense(num_classes,activation='softmax',)(c1) # kernel_constraint=max_norm(.5)
    return models.Model(inp,c1) 

    

def ShallowConvNetMultiClass(input_shape,network_params):
    """
    input_shape: (time,channels,1)
    Riedmueller 2015 overfitted
    """
    use_bn=network_params['use_bn']
    do_ratio=network_params['do_ratio']
    num_classes = network_params['num_classes']
    
    inp = layers.Input(input_shape)
    numChannels = K.int_shape(inp)[2]
    print('numChannels',numChannels)
    
    # 
    c1 = layers.Conv2D(32,kernel_size=(5,1),strides=(2,1),padding='same',use_bias=~use_bn,name='temporal_conv_relu')(inp)
    if use_bn:
        c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(32,kernel_size=(13,1),padding='same',use_bias=True,name='temporal_conv')(c1)
    c1 = layers.Conv2D(32,kernel_size=(1,numChannels),padding='valid',use_bias=True,name='spatial_conv')(c1)    
    # square
    c1 = layers.Activation(K.square)(c1)
    # (1,35 |7)
    c1 = layers.AveragePooling2D(pool_size=(25,1),strides=(7,1))(c1)
    # not existent
    #c1 = layers.Conv2D(128,kernel_size=(1,13),padding='same', kernel_constraint=max_norm(2.),use_bias=True)(c1)
    c1 = layers.Activation(K.log)(c1)
    
    c1 = layers.Flatten()(c1)
    c1 = layers.Dropout(do_ratio)(c1)
    if num_classes==2:
        c1 = layers.Dense(1,activation='sigmoid')(c1)
    else:
        c1 = layers.Dense(num_classes,activation='softmax',)(c1) # kernel_constraint=max_norm(.5)
    return models.Model(inp,c1) 



def ShallowConvNetSTFT(input_shape,use_bn=False,do_ratio=.0):
    """
    Schirrmeister R T, 2017
    """
    # input_shape: (time_steps,frequencies,N_SPHARA_harmonics)
    numChannels = input_shape[-1]
    inp = layers.Input(input_shape)
    c1 = layers.Lambda(lambda x: K.expand_dims(x,1))(inp)
    c1 = layers.Permute([4,2,3,1])(c1)
    # new shape: N_SPHARA_harmonics, time_steps,frequencies ,features
    numChannels = K.int_shape(c1)[1]
    c1 = layers.Conv3D(40,kernel_size=(1,3,1),padding='same', kernel_constraint=max_norm(2.),use_bias=True,activation='relu')(c1)
    c1 = layers.Conv3D(40,kernel_size=(1,1,8),padding='same', kernel_constraint=max_norm(2.),use_bias=True,activation='relu')(c1)
    c1 = layers.Conv3D(40,kernel_size=(1,1,8),padding='same', kernel_constraint=max_norm(2.),use_bias=True,activation='relu')(c1)
    c1 = layers.Conv3D(40,kernel_size=(numChannels,1,1),padding='valid', kernel_constraint=max_norm(2.),use_bias=True,activation='relu')(c1)    
    if use_bn:
        c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.AveragePooling3D(pool_size=(1,2,1),strides=(1,2,1))(c1)
    c1 = layers.Activation('relu')(c1)
    
    print('numChannels',numChannels)
    
    c1 = layers.Flatten()(c1)
    c1 = layers.Dropout(do_ratio)(c1)
    c1 = layers.Dense(1,activation='sigmoid', kernel_constraint=max_norm(.5))(c1)
    return models.Model(inp,c1) 
print('r')
def MultipleStreamNet(input_shape,network_params):
    """
    Split Left,Right and middle z Brain into multiple streams
    -> Not converging
    """
    leftChannelsIdx  = network_params['leftChannels']
    rightChannelsIdx = network_params['rightChannels']
    midChannelsIdx   = network_params['midChannels']
    use_bn           = network_params['use_bn']
    inp = layers.Input(input_shape)
    c1 = layers.Lambda(lambda x: K.expand_dims(x,1))(inp)
    c1 = layers.Permute([3,2,1])(c1)
    # new shape: N_SPHARA_harmonics, time_steps, features
    
    c1 = layers.Conv2D(32,kernel_size=(1,32),strides=(1,1),padding='same')(c1)
    if use_bn:
        c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    
    leftChannels = layers.Lambda(lambda x: tf.gather(x,leftChannelsIdx,axis=1))(c1)
    rightChannels = layers.Lambda(lambda x: tf.gather(x,rightChannelsIdx,axis=1))(c1)
    midChannels = layers.Lambda(lambda x: tf.gather(x,midChannelsIdx,axis=1))(c1)
    
    leftChannels = layers.Conv2D(32,kernel_size=(1,32),padding='same')(leftChannels)
    rightChannels = layers.Conv2D(32,kernel_size=(1,32),padding='same')(rightChannels)
    midChannels = layers.Conv2D(32,kernel_size=(1,32),padding='same')(midChannels)
    
    leftChannels = layers.Conv2D(64,kernel_size=(len(leftChannelsIdx),1),padding='valid')(leftChannels)
    rightChannels = layers.Conv2D(64,kernel_size=(len(rightChannelsIdx),1),padding='valid')(rightChannels)
    midChannels = layers.Conv2D(64,kernel_size=(len(midChannelsIdx),1),padding='valid')(midChannels)
    
    # nonlinear 
    leftChannels = layers.Conv2D(64,kernel_size=(1,32),padding='same',activation='relu')(leftChannels)
    rightChannels = layers.Conv2D(64,kernel_size=(1,32),padding='same',activation='relu')(rightChannels)
    midChannels = layers.Conv2D(64,kernel_size=(1,32),padding='same',activation='relu')(midChannels)
    
    leftChannels = layers.MaxPooling2D(pool_size=(1,4),strides=(1,4))(leftChannels)
    rightChannels = layers.MaxPooling2D(pool_size=(1,4),strides=(1,4))(rightChannels)
    midChannels = layers.MaxPooling2D(pool_size=(1,4),strides=(1,4))(midChannels)
    
    conc = layers.Concatenate(axis=1)([leftChannels,midChannels,rightChannels])
    c1 = layers.Conv2D(32,kernel_size=(1,32),activation='relu',padding='same',strides=(1,1))(conc)
    c1 = layers.Flatten()(c1)
    c1 = layers.Dense(1,activation='sigmoid')(c1)
    
    
    #c1Left = EEGNetSpecial(c1,F2,use_bn,do_ratio)
    #c1Right = EEGNetSpecial(c1,F2,use_bn,do_ratio)
    #c1Mid = EEGNetSpecial(c1,F2,use_bn,do_ratio)
    
    
    return models.Model(inp,c1)


def filterPredictionModuleDirect(x,numChannels,numBeamCoefficients):
    """
    Neural Network Adaptive Beamforming forRobust Multichannel Speech Recognition
    """
    shared = layers.LSTM(64,return_sequences=True)(x)
    shared = layers.LSTM(64,return_sequences=True)(shared)
    linearList = []
    for ch in range(numChannels):
        flat = layers.Flatten()(shared)
        linear = layers.Dense(numBeamCoefficients)(flat)
        linearList+=[linear]
    return linearList

def filterPredictionModuleShared(x,numChannels,numBeamCoefficients):
    """
    Neural Network Adaptive Beamforming forRobust Multichannel Speech Recognition
    They use a first 512 cell LSTM followed by C 256 cell LSTM's for each channel
    """
    shared = layers.LSTM(32,return_sequences=True)(x)
    # Heavy computational effort, when using numChannel LSTM's
    lstmList = []
    for ch in range(numChannels):
        lstm = layers.LSTM(16,return_sequences=True)(shared)
        linear = layers.TimeDistributed(layers.Dense(numBeamCoefficients))(lstm)
        lstmList+=[linear]
    return lstmList
#function that performs the convolution

    
def NeuralBeamformer(input_shape,network_params):
    """
    Implements Adaptive Filter and Sum Beamformer via LSTM's with implicit delay in filter coefficients.
    """
    batch_size = network_params['batch_size']
    frame_size = network_params['frame_size']
    use_bn     = network_params['use_bn']
    numBeamCoefficients = network_params['numBeamCoefficients']
    numChannels = input_shape[-1]
    def sample_wise_conv1d(x_filters):
        """
        This is very tricky and we're getting help from K.depthwise_conv2d (the only convolution that treats channels individually), 
        where we will transform the samples into channels, have the desired outputs per channel and then reshape back to the expected
        """
        #reshape and reorder inputs properly
        padding='same'
        x = x_filters[0]
        filters = x_filters[1]
        
        x= K.expand_dims(x, axis=0)
        x = K.permute_dimensions(x, (0,2,3,1))
        #print('in shape', K.int_shape(x))

        #perform the convolution
        #Have filters with shape (kernel_size, input_dim, batch_size, output_dim)
        filters = K.expand_dims(filters,axis=-1)
        filters = K.permute_dimensions(filters,(1,2,0,3))
        #print("filter shape", K.int_shape(filters))
        # depthwise_conv2d is not part of the backend and of tf in tf.__version__: 2.0, but in 2.2
        results =  K.depthwise_conv2d(x, filters,padding=padding,strides=(1,1))
        #print('out shape', K.int_shape(results))
        
        #keep track of output length for reshaping
        if padding=='valid':
            outlen = frame_size - filtersize + 1
        elif padding=='same':
            outlen = frame_size
        out_dim = 1
        #reshape and reorder the results properly
        results = K.reshape(results, (outlen, batch_size, out_dim))
        results = K.permute_dimensions(results, (1,0,2))
        #print('final shape', K.int_shape(results))
        return results
    
    inp = layers.Input(input_shape)
    c1 = layers.AveragePooling1D(pool_size=2,strides=2)(inp)
    filterCoefficientsList = filterPredictionModuleDirect(c1,numChannels,numBeamCoefficients=numBeamCoefficients)
    


    FilteredSignals = []
    for k in range(numChannels):
        coefficients = layers.Lambda(lambda x:tf.expand_dims(x,axis=-1))(filterCoefficientsList[k])
        InputChannel = layers.Lambda(lambda x: x[:,:,k:k+1])(inp)
        #https://stackoverflow.com/questions/54509341/keras-1d-convolutions-with-different-filter-for-each-sample-in-mini-batch
        FilteredSignals+=[layers.Lambda(sample_wise_conv1d)([InputChannel,coefficients])]
    
    newSignal = layers.Add()(FilteredSignals)#
    newSignal = layers.Lambda(lambda x:tf.expand_dims(x,axis=-1))(newSignal)
    print(K.int_shape(newSignal))
    c1 = layers.Conv2D(32,kernel_size=(64,1),strides=(1,1))(newSignal)
    if use_bn:
        c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.MaxPooling2D(pool_size=(4,1),strides=(4,1))(c1)
    
    c1 = layers.Conv2D(64,kernel_size=(32,1),strides=(1,1))(newSignal)
    if use_bn:
        c1 = layers.BatchNormalization()(c1)
    c1 = layers.MaxPooling2D(pool_size=(4,1),strides=(4,1))(c1)
    c1 = layers.Activation('relu')(c1)
    
    
    flat = layers.Flatten()(c1)
    clf = layers.Dense(1,activation='sigmoid')(flat)
    m = models.Model(inp,clf,name='NeuralBeamformer')
    return m
    
def eegBeamFormer(input_shape,network_params):
    """
    Idea: Channels contain correlated signal s and can be used as an 'ensemble to reduce the noise n. x = s+n
    The network must select appropriate channels, which share the same signal and shift them.
    The similarity can be found in a naive way by cross correlation with a filter bank.
    
    The selection whether a channel contains the signal s or not can be decided by a sigmoid function.
    If the signal is present in channel j, then we can find the delay by finding the maximum correlation on the time axis (Maxpooling).
    The reference Signal should have its maximum correlation at 0 ->  edge effects?-> how can other signals, what if the signal in 
    another channel start before the signal in the reference channel i.e. s_j(t+tau)=s_ref(t)?.
    
    Maybe it is better to shift all signals (including the reference signal) to another time point.
    """
    # 0 degree reference channel, all other channels should be aligned to it
    referencChannel = network_params['ref_ch']
    
    inp = layers.Input(input_shape)
    c1 = layers.Lambda(lambda x: K.expand_dims(x,1))(inp)
    c1 = layers.Permute([3,2,1])(c1)
    # new shape: N_SPHARA_harmonics, time_steps, features
    numChannels = K.int_shape(c1)[1]
    timeSteps  = K.int_shape(c1)[2]
    # large kernel with same padding -> many '0' get padded
    c1 = layers.Conv2D(64,kernel_size=(1,128),padding='same',strides=(1,1))(c1)
    # sets all but the maximum value to 0
    # https://stackoverflow.com/questions/59229131/zero-out-everything-except-for-max-in-custom-keras-layer
    out = layers.Lambda(lambda x: tf.expand_dims(tf.cast(tf.equal(x, tf.reduce_max(x)), tf.float32)*x,axis=-1))(c1)

    
    return models.Model(inp,shifted)

    
def classifierLeftRightMovement(encoder):
    xin = encoder.output
    xin = layers.Flatten()(xin)
    out = layers.Dense(1,activation='sigmoid')(xin)
    m = models.Model(encoder.input,out)
    return m

def up_transition(x,nb_filter,stage):
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_T'
    relu_name_base = 'relu' + str(stage) + '_T'
    trans_name_base = 'tran' + str(stage) 

    x = layers.BatchNormalization(name=conv_name_base+'_bn')(x)
    #x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = layers.Activation('relu', name=relu_name_base)(x)
    
    x  = layers.Conv2D(nb_filter*2,kernel_size=(1,1), name=conv_name_base,use_bias=False)(x)
    
    x  = layers.Conv2DTranspose(nb_filter,kernel_size=(1,3),strides=(1,2),padding='same',name=trans_name_base)(x)
    return x

def makeDENSEdecoder2(x, nb_filter,growth_rate = -4):
    isModel=True
    
    if isModel:
        xin = x.output
    #enc = layers.Input(input=x)
    up1 = up_transition(xin,nb_filter,'up1')
     #dense_block takes: (x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True)
    up1, nb_filter = dense_block(up1, 'up1', nb_layers=4, nb_filter=nb_filter, growth_rate=growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True)
    
    #up1  = layers.Conv2DTranspose(nb_filter,kernel_size=(1,3),strides=(1,2),padding='same')(up1)
    up1 = up_transition(up1,nb_filter,'up2')
    #up1  = layers.BatchNormalization()(up1)
    
    up1, nb_filter = dense_block(up1, 'up2', nb_layers=6, nb_filter=nb_filter, growth_rate=growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True)
    
    up1 = up_transition(up1,nb_filter,'up3')
    up1, nb_filter = dense_block(up1, 'up3', nb_layers=4, nb_filter=nb_filter, growth_rate=growth_rate)
    
    up1 = up_transition(up1,nb_filter,'up4')
    up1, nb_filter = dense_block(up1, 'up4', nb_layers=2, nb_filter=nb_filter, growth_rate=growth_rate)
    
    #up1  = layers.Conv2D(nb_filter*2,kernel_size=(1,1), name='last1x1',use_bias=False)(up1)
    up1  = layers.BatchNormalization(axis=-1)(up1)
    out = layers.Conv2D(1,kernel_size=(1,1),activation='sigmoid')(up1)
    
    m = models.Model(x.input,out)
    return m    

def EEGNetSpecial(c1,F2,use_bn,do_ratio):
    if use_bn:
        c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('elu')(c1)
    c1 = layers.AveragePooling2D(pool_size=(1,4),strides=(1,4),padding='same')(c1)
    c1 = layers.Dropout(do_ratio)(c1)
    # Depthwise Separable Convolutions
    # https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728

    c1 = layers.SeparableConv2D(F2,kernel_size=(1,16),padding='same', depth_multiplier=1)(c1)
    if use_bn:
        c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('elu')(c1)
    c1 = layers.AveragePooling2D(pool_size=(1,8),strides=(1,8))(c1)
    c1 = layers.Dropout(do_ratio)(c1)
    c1 = layers.Flatten()(c1)
    return c1

def EEGNet(input_shape,network_params={'F1':32,'F2':32,'do_ratio':.25,'depth_multi':4,'use_bn':False,'kernel_size1':256}):
    """
    Implemented according to their paper: Lawhern, 2018
    Their settings:
    Only 22 electrodes, sampled at 256Hz ->bandpass filtered .5:100Hz-> resampled to 128Hz
    -> They use 1second as input time -> 128 time_steps
    overfitted, also overfits when mixed subjects
    can also be found at: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
    """
    F1 = network_params['F1']
    F2 = network_params['F2']
    depth_multi = 3#network_params['depth_multi']
    kernel_size1 = 5#network_params['kernel_size1']
    if F2=='match':
        F2 = F1*depth_multi
    elif F2 =='compressed':
        F2 = F1*int(depth_multi//2)
    elif F2=='overcomplete':
        F2= F1*2*depth_multi
    elif type(F2)==int:
        pass
    else:
        raise ValueError('F2 must be one of match, compressed '
                 'or overcomplete, passed as a string or an integer.')
    do_ratio    = network_params['do_ratio']
    use_bn = network_params['use_bn']
    #fs = network_params['fs']
    # input_shape: (time_steps,N_SPHARA_harmonics)
    numChannels = input_shape[-1]
    inp = layers.Input(input_shape)
    c1 = layers.Lambda(lambda x: K.expand_dims(x,1))(inp)
    c1 = layers.Permute([3,2,1])(c1)
    # new shape: N_SPHARA_harmonics, time_steps, features
    c1 = layers.Conv2D(F1,kernel_size=(1,kernel_size1),padding='same')(c1)
    c1 = layers.Conv2D(depth_multi*F1,kernel_size=(numChannels,1),padding='valid', kernel_constraint=max_norm(1.))(c1)
    
    c1 = EEGNetSpecial(c1,F2,use_bn,do_ratio)
    
    c1 = layers.Dense(1,activation='sigmoid')(c1)#,kernel_constraint=max_norm(.5)
    return models.Model(inp,c1)



def EEGNetLawhern(input_shape, 
             dropoutRate = 0.5, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, 
             dropoutType = 'Dropout',
             network_params={}):
    #(nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5, kernLength = 64, F1 = 8,  D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout')
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 
    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """
    Chans = input_shape[1]
    Samples = input_shape[0]
    kernLength = 64
    use_bn=network_params['use_bn']
    do_ratio=network_params['do_ratio']
    num_classes = network_params['num_classes']
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = layers.Input(shape = input_shape)
    block1  = layers.Permute([2,1,3])(input1 )
    ##################################################################
    block1       = layers.Conv2D(F1, (1, kernLength), padding = 'same',
                                   use_bias = False)(block1)
    block1       = layers.BatchNormalization()(block1)
    block1       = layers.DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = layers.BatchNormalization()(block1)
    block1       = layers.Activation('elu')(block1)
    block1       = layers.AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = layers.SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = layers.BatchNormalization()(block2)
    block2       = layers.Activation('elu')(block2)
    block2       = layers.AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = layers.Flatten(name = 'flatten')(block2)
    
    dense        = layers.Dense(num_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = layers.Activation('softmax', name = 'softmax')(dense)
    
    return models.Model(inputs=input1, outputs=softmax)
    
def makeLSTM(input_shape):
    # only 1 channel can be processed
    inp   = layers.Input(input_shape)
    lstm1 = layers.LSTM(8)(inp)
    return models.Model(inp,lstm1)
    
    
class CustomModel(tf.keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        output={m.name: m.result() for m in self.metrics[:-1]}
        if 'confusion_matrix_metric' in self.metrics_names:
            self.metrics[-1].fill_output(output)
        return output
    
    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        
        x, y = data
        y_pred = self(x, training=False)  # Forward pass
        # Compute the loss value.
        # The loss function is configured in `compile()`.
        loss = self.compiled_loss(
            y,
            y_pred,
            regularization_losses=self.losses,
        )

        self.compiled_metrics.update_state(y, y_pred)
        output={m.name: m.result() for m in self.metrics[:-1]}
        if 'confusion_matrix_metric' in self.metrics_names:
            self.metrics[-1].fill_output(output)    
        return output
