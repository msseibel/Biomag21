from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def car():
    """
    """
    eegdata = eegdata[:,start_t:stop_t]-np.mean(eegdata[:,start_t:stop_t],axis=0,keepdims=True)
    return eegdata
def filtering(eegdata,f_cut,fs,filt_type='high',debug_filter=False,use0phase=False):
    """
    : eegdata (channels,time_steps)
    :param f_cut: -3dB frequency in Hz
    """
    # correction for -3dB at cut off, only valid for butterworth
    # https://github.com/scipy/scipy/issues/9371
    # TBH: I just copied the results of the discussion
    #if use0phase and filt_type=='low':
    #    f_cut = f_cut / 0.8
    #elif use0phase and filt_type=='high':
    #    f_cut = f_cut * 0.8
    
    if not fs:
        sos  = signal.butter(8, f_cut,btype=filt_type,analog=True, output='sos')
    else:
        sos  = signal.butter(8, Wn=f_cut,btype=filt_type,fs=fs, output='sos')#/(fs/2)

    if use0phase:
        y = signal.sosfiltfilt(sos,  eegdata)
    else:
        y = signal.sosfilt(sos,  eegdata)
       
    return y
 

    


