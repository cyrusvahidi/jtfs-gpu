'''
Copyright (c) Baptiste Caramiaux, Etienne Thoret
All rights reserved

'''
import numpy as np
import math
from scipy import signal
import utils
#from lib import utils
import features  #import spectrum2scaletime, scaletime2scalerate, scalerate2cortical, waveform2auditoryspectrogram
import matplotlib.pylab as plt



def load_static_params():
    strf_params = {}
    strf_params['scales'] = [
        0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66, 8.00
    ]
    strf_params['rates'] = [
        -32, -22.6, -16, -11.3, -8, -5.70, -4, -2, -1, -.5, -.25, .25, .5, 1, 2, 4,
        5.70, 8, 11.3, 16, 22.6, 32
    ]
    strf_params['sr_time'] = 250
    return strf_params

def load_strf_params(rates = [-32, -22.6, -16, -11.3, -8, -5.70, -4, -2, -1, -.5, -.25, .25, .5, 1, 2, 4,5.70, 8, 11.3, 16, 22.6, 32], 
                     scales = [0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66, 8.00], sr_time=250):
    strf_params = {}
    strf_params['scales'] = scales
    strf_params['rates'] = rates
    strf_params['sr_time'] = sr_time
    return strf_params


def spectrogram(wavtemp,
                audio_fs=44100,
                duration=0.25,
                duration_cut_decay=0.05,
                resampling_fs=16000,
                sr_time=250,
                offset=0.0):
    auditory_params = load_static_params()
    # resampling_fs = auditory_params['newFs']
    # duration = auditory_params['duration']
    # duration_cut_decay = auditory_params['duration_cut_decay']
    sr_time = auditory_params['sr_time']
    wavtemp = np.r_[wavtemp, np.zeros(resampling_fs)]
    print(resampling_fs)
    if duration==-1: 
        print('no duration cut')
    elif wavtemp.shape[0] > math.floor(duration * audio_fs):
        offset_n = int(offset * audio_fs)
        duration_n = int(duration * audio_fs)
        duration_decay_n = int(duration_cut_decay * audio_fs)
        wavtemp = wavtemp[offset_n:offset_n + duration_n]
        if offset_n==0:
            wavtemp[wavtemp.shape[0] - duration_decay_n:] = wavtemp[
                wavtemp.shape[0] - duration_decay_n:] * utils.raised_cosine(
                    np.arange(duration_decay_n), 0, duration_decay_n)
        else:
            wavtemp[wavtemp.shape[0] - duration_decay_n:] = wavtemp[
                wavtemp.shape[0] - duration_decay_n:] * utils.raised_cosine(
                    np.arange(duration_decay_n), 0, duration_decay_n)   
            wavtemp[:duration_decay_n] = wavtemp[:duration_decay_n] * utils.raised_cosine(
                    np.arange(duration_decay_n), duration_decay_n, duration_decay_n)

    
    wavtemp = (wavtemp / 1.01) / (np.max(wavtemp) + np.finfo(float).eps)

    wavtemp = signal.resample(wavtemp,
                              int(wavtemp.shape[0] / audio_fs * resampling_fs))

    waveform2auditoryspectrogram_args = {
        'frame_length':
        1000 / sr_time,  # sample rate 125 Hz in the NSL toolbox
        'time_constant': 8,
        'compression_factor': -2,
        'octave_shift': math.log2(resampling_fs / resampling_fs),
        'filt': 'p',
        'VERB': 0
    }

    auditory_spectrogram_ = features.waveform2auditoryspectrogram(
        wavtemp, **waveform2auditoryspectrogram_args)
    return auditory_spectrogram_


def spectrum(wavtemp,
             audio_fs=44100,
             duration=0.25,
             duration_cut_decay=0.05,
             resampling_fs=16000,
             sr_time=250,
             offset=0):
    auditory_spectrogram_ = spectrogram(wavtemp, audio_fs, duration,
                                        duration_cut_decay, resampling_fs,
                                        sr_time, offset)
    auditory_spectrum_ = np.mean(auditory_spectrogram_, axis=0)
    return auditory_spectrum_


def mps(wavtemp,
        audio_fs=44100,
        duration=0.25,
        duration_cut_decay=0.05,
        resampling_fs=16000,
        sr_time=250,
        offset=0):
    auditory_spectrogram_ = spectrogram(wavtemp, audio_fs, duration,
                                        duration_cut_decay, resampling_fs,
                                        sr_time, offset)
    strf_args = {
        'num_channels': 128,
        'num_ch_oct': 24,
        'sr_time': sr_time,
        'nfft_rate': 2 * 2**utils.nextpow2(auditory_spectrogram_.shape[0]),
        'nfft_scale': 2 * 2**utils.nextpow2(auditory_spectrogram_.shape[1]),
        'KIND': 2
    }
    # Spectro-temporal modulation analysis
    # Based on Hemery & Aucouturier (2015) Frontiers Comp Neurosciences
    # nfft_fac = 2  # multiplicative factor for nfft_scale and nfft_rate
    # nfft_scale = nfft_fac * 2**utils.nextpow2(auditory_spectrogram_.shape[1])
    mod_scale, phase_scale, _, _ = features.spectrum2scaletime(
        auditory_spectrogram_, **strf_args)
    mps_, phase_scale_rate, _, _ = features.scaletime2scalerate(
        mod_scale * np.exp(1j * phase_scale), **strf_args)
    # repres = repres[:, :int(repres.shape[1] / 2)]
    return mps_


def strf(wavtemp,
         audio_fs=44100,
         duration=0.25,
         duration_cut_decay=0.05,
         resampling_fs=16000,
         sr_time=250,
         offset=0, 
         rates=[-32, -22.6, -16, -11.3, -8, -5.70, -4, -2, -1, -.5, -.25, .25, .5, 1, 2, 4,5.70, 8, 11.3, 16, 22.6, 32], 
         scales=[0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66, 8.00]):

    auditory_spectrogram_ = spectrogram(wavtemp, audio_fs, duration,
                                        duration_cut_decay, resampling_fs,
                                        sr_time, offset)

    auditory_params = load_strf_params(rates, scales, sr_time)
    scales = auditory_params['scales']
    rates = auditory_params['rates']

    strf_args = {
        'num_channels': 128,
        'num_ch_oct': 24,
        'sr_time': sr_time,
        'nfft_rate': 2 * 2**utils.nextpow2(auditory_spectrogram_.shape[0]),
        'nfft_scale': 2 * 2**utils.nextpow2(auditory_spectrogram_.shape[1]),
        'KIND': 2
    }
    # Spectro-temporal modulation analysis
    # Based on Hemery & Aucouturier (2015) Frontiers Comp Neurosciences
    # nfft_fac = 2  # multiplicative factor for nfft_scale and nfft_rate
    # nfft_scale = nfft_fac * 2**utils.nextpow2(stft.shape[1])
    mod_scale, phase_scale, _, _ = features.spectrum2scaletime(
        auditory_spectrogram_, **strf_args)

    # Scales vs. Time => Scales vs. Rates
    # nfft_rate = nfft_fac * 2**utils.nextpow2(stft.shape[0])
    scale_rate, phase_scale_rate, _, _ = features.scaletime2scalerate(
        mod_scale * np.exp(1j * phase_scale), **strf_args)
    # print(scale_rate.shape)
    # print(phase_scale_rate.shape)
    #num_channels, num_ch_oct, sr_time, nfft_rate, nfft_scale)
    strf_ = features.scalerate2cortical(auditory_spectrogram_, scale_rate,
                                        phase_scale_rate, scales, rates,
                                        **strf_args)
    # print(strf_.shape)
    #num_ch_oct, sr_time, nfft_scale, nfft_rate, 2)
    return strf_, auditory_spectrogram_, mod_scale, scale_rate


if __name__ == "__main__":
    audio, fs = utils.audio_data(
        '/Users/baptistecaramiaux/Work/Projects/TimbreProject_Thoret/Code\ and\ data/timbreStudies/ext/sounds/Iverson1993Whole/01.W.Violin.aiff'
    )
    spectrum(audio, fs)
