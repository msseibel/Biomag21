import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from libs import utils

class Base(object):
    def __init__(self, base_arg0, base_arg1, base_arg2):
        self.base_arg0 = base_arg0

class Derived(Base):
    def __init__(self, derived_arg, *args):
        super().__init__(*args)

class StatFunc(object):
    """
    Base class for the definition of sliding stats.
    """
    def __init__(self, Twin: int,fs: int):
        """
        Twin int or array_like structur
        """
        self.Twin = Twin
        self.fs = fs

class SlidingStd(StatFunc):
    """
    Splits signal in samples/Twin sequences.
    """
    def __init__(self, *args):
        super().__init__(*args)

    def __call__(self,signal):
        """
        Input:  signal: (channels,samples)
        Output: signal: (channels,samples/Twin)
        """
        (channels,samples)= signal.shape
        taxis = np.arange(0,samples,self.Twin)/self.fs 
        
        # eventually we can just pass this line to the parent class
        K = int(samples/self.Twin)
        signal = np.std(signal.reshape(channels,K,self.Twin),axis=-1)
        return signal,taxis
        
        
class SlidingMean(StatFunc):
    """
    Splits signal in samples/Twin sequences.
    """
    def __init__(self, *args):
        super().__init__(*args)

    def __call__(self,signal):
        """
        Input:  signal: (channels,samples)
        Output: signal: (channels,samples/Twin)
        """
        (channels,samples)= signal.shape
        taxis = np.arange(0,samples,self.Twin)/self.fs 
        
        # eventually we can just pass this line to the parent class
        K = int(samples/self.Twin)
        signal = np.mean(signal.reshape(channels,K,self.Twin),axis=-1)
        return signal,taxis

def bandpower(x, fs, fmin, fmax):
    f, Pxx = signal.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])


def simpleFeatures(recording,infotest,fs,is_meg=False):
    """
    recording: meg,typ2record,type3record
    """
    df = 1/recording.shape[1]*fs#freq_axis[1]
    names = ['ch'+str(idx) for idx in range(len(recording))]
    numUniques = [len(np.unique(ch))for ch in recording]
    stds = np.std(recording,axis=-1)
    means = np.mean(recording,axis=-1)
    mins = np.min(recording,axis=-1)
    maxs = np.max(recording,axis=-1)
    recordingfft = np.fft.fft(recording-means.reshape(recording.shape[0],1))
    left = int(recording.shape[1]/2)
    recordingfft = (np.abs(recordingfft[:,:left]/left)**2)/2
    
    skew = stats.skew(recording,axis=-1)
    energy = np.sum(recordingfft,axis=1)
    cumsums = [np.cumsum(feat) for feat in recordingfft]

    # division by signal_length equals multiplication by df s.t. we convert idx to frequency
    f50 = [np.where(cumsums[ch]>energy[ch]*0.5)[0][0]/(5*60) for ch in range(len(recording))]
    f95 = [np.where(cumsums[ch]>energy[ch]*0.95)[0][0]/(5*60) for ch in range(len(recording))]
    
    E10   = [np.mean(ch[int(0/df):int(10/df)]) for ch in recordingfft]
    E1020 = [np.mean(ch[int(10/df):int(20/df)]) for ch in recordingfft]
    E2030 = [np.mean(ch[int(20/df):int(30/df)]) for ch in recordingfft]
    EcHPI = [np.mean(ch[int(300/df):int(340/df)]) for ch in recordingfft]
    E220280  = [np.mean(ch[int(260/df):int(300/df)]) for ch in recordingfft]
    E340400  = [np.mean(ch[int(300/df):int(340/df)]) for ch in recordingfft]
    
    if recording.shape[0]==160 or is_meg:
        fids = utils.load_key_chain(infotest['D'],['fiducials','fid','pnt'])
        chanpos = utils.load_key_chain(infotest['D'],['sensors','meg','chanpos'])
        dist_nas  = [np.linalg.norm(fids[0].reshape(1,3)-chanpos,axis=-1)[ch] for ch in range(160)]
        dist_lpa  = [np.linalg.norm(fids[1].reshape(1,3)-chanpos,axis=-1)[ch] for ch in range(160)]
        dist_rpa  = [np.linalg.norm(fids[2].reshape(1,3)-chanpos,axis=-1)[ch] for ch in range(160)]
        dist_mrk4 = [np.linalg.norm(fids[3].reshape(1,3)-chanpos,axis=-1)[ch] for ch in range(160)]
        dist_mrk5 = [np.linalg.norm(fids[4].reshape(1,3)-chanpos,axis=-1)[ch] for ch in range(160)]
    else:
        dist_nas  = np.nan
        dist_lpa  = np.nan
        dist_rpa  = np.nan
        dist_mrk4 = np.nan
        dist_mrk5 = np.nan
    df = pd.DataFrame({'#uniqueValues':numUniques,'std':stds,'mean':means,'skew':skew,
                       'min':mins,'max':maxs,'f50':f50,'f95':f95,'energy':energy,'E10':E10,
                      'E1020':E1020,'E2030':E2030,'EcHPI':EcHPI,'E260300':E220280,
                       'E300340':E340400,'dist_nas':dist_nas,'dist_lpa':dist_lpa,
                       'dist_rpa':dist_rpa,'dist_mrk4':dist_mrk4,'dist_mrk5':dist_mrk5
                      },index=names)
    df['energy'] = df['energy'].map('{:,.3e}'.format)
    df['E10'] = df['E10'].map('{:,.3e}'.format)
    df['E1020'] = df['E1020'].map('{:,.3e}'.format)
    df['E2030'] = df['E2030'].map('{:,.3e}'.format)
    #display(df)
    return df

def plotRecordingsFFT(recording,title,fs,indB=False,columns=4):
    recordingfft = np.abs(np.fft.fft(recording-recording.mean(axis=-1,keepdims=True)))
    left = int(recording.shape[1]/2)-1
    recordingfft = (np.abs(recordingfft[:,:left]/left)**2)/2
    
    rows = int(np.ceil(recordingfft.shape[0]/columns))
    
    fig,axes = plt.subplots(rows,columns,figsize=(int(columns*4),int(4*rows*1.2)))
    freq_axis = np.fft.fftfreq(left*2,1/fs)[:left] 
    axes = axes.flatten()
    names = ['ch'+str(idx) for idx in range(len(recordingfft))]
    for idx,ax in enumerate(axes):
        if idx>=recordingfft.shape[0]:
            break
        if indB:
            ax.plot(freq_axis,20*np.log10(1e-7+np.abs(recordingfft[idx])))
            ax.set_ylabel('20log|X(f)|')
        else:
            ax.plot(freq_axis,(np.abs(recordingfft[idx])))
            ax.set_ylabel('X(f))')
        ax.set_title(names[idx])
        ax.set_xlabel('f [Hz]')
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return axes