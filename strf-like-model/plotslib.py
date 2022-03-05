'''
Copyright (c) Baptiste Caramiaux, Etienne Thoret
All rights reserved

'''

import matplotlib.pylab as plt
import auditory
import utils
import pickle
import numpy as np
import scipy.io as sio


def strf2avgvec(strf):
    strf_scale_rate = np.mean(np.abs(strf), axis=(0,1))
    strf_freq_rate  = np.mean(np.abs(strf), axis=(0,2))
    strf_freq_scale = np.mean(np.abs(strf), axis=(0,3))
    avgvec = np.concatenate((np.ravel(strf_scale_rate), np.ravel(strf_freq_rate), np.ravel(strf_freq_scale)))
    return avgvec

def avgvec2strfavg(avgvec,nbChannels=128,nbRates=22,nbScales=11):
    idxSR = nbRates*nbScales
    idxFR = nbChannels*nbRates
    idxFS = nbChannels*nbScales
    strf_scale_rate = np.reshape(avgvec[:idxSR],(nbScales,nbRates))
    strf_freq_rate = np.reshape(avgvec[idxSR:idxSR+idxFR],(nbChannels,nbRates))
    strf_freq_scale = np.reshape(avgvec[idxSR+idxFR:],(nbChannels,nbScales))
    return strf_scale_rate, strf_freq_rate, strf_freq_scale

def plotStrfavg(strf_scale_rate, strf_freq_rate, strf_freq_scale,aspect_='auto', interpolation_='none',figname='defaut',show='true'):
    plt.suptitle(figname, fontsize=10)
    plt.subplot(1,3,1)
    plt.imshow(strf_scale_rate, aspect=aspect_, interpolation=interpolation_,origin='lower')
    plt.xlabel('Rates (Hz)', fontsize=10)
    plt.ylabel('Scales (c/o)', fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,2)
    plt.imshow(strf_freq_rate, aspect=aspect_, interpolation=interpolation_,origin='lower')
    plt.xticks([])
    plt.yticks([]) 
    plt.xlabel('Rates (Hz)', fontsize=10)
    plt.ylabel('Frequencies (Hz)', fontsize=10)    
    plt.subplot(1,3,3)
    plt.imshow(strf_freq_scale, aspect=aspect_, interpolation=interpolation_,origin='lower')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Scales (c/o)', fontsize=10)
    plt.ylabel('Frequencies (Hz)', fontsize=10)
    plt.savefig(figname+'.png')
    if show=='true':
        plt.show()


def plotStrfavgEqual(strf_scale_rate, strf_freq_rate, strf_freq_scale,aspect_='equal', interpolation_='none',figname='defaut',cmap='jet'):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    plt.subplot(1,3,1)
    im = plt.imshow(strf_scale_rate, aspect=strf_scale_rate.shape[1]/strf_scale_rate.shape[0], interpolation=interpolation_,origin='lower',cmap=cmap)
    # plt.xlabel('Rate (Hz)', fontsize=10)
    # plt.ylabel('Scale (c/o)', fontsize=10)
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')   
    plt.xlabel('Rates (Hz)', fontsize=10)
    plt.ylabel('Scales (c/o)', fontsize=10)    

    plt.subplot(1,3,2)
    im = plt.imshow(strf_freq_rate, aspect=strf_freq_rate.shape[1]/strf_freq_rate.shape[0], interpolation=interpolation_,origin='lower',cmap=cmap)
    plt.xticks([])
    plt.yticks([]) 
    # plt.axis('off')
    plt.xlabel('Rates (Hz)', fontsize=10)
    plt.ylabel('Frequencies (Hz)', fontsize=10)   

    # plt.xlabel('Rate (Hz)', fontsize=10)
    # plt.ylabel('Frequency (Hz)', fontsize=10)    
    plt.subplot(1,3,3)
    im = plt.imshow(strf_freq_scale, aspect=strf_freq_scale.shape[1]/strf_freq_scale.shape[0], interpolation=interpolation_,origin='lower',cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')   
    plt.xlabel('Scales (c/o)', fontsize=10)
    plt.ylabel('Frequencies (Hz)', fontsize=10)


    plt.savefig(figname+'.png',bbox_inches='tight')
    plt.show() 
        
