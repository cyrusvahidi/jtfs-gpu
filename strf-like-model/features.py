'''
Copyright (c) Baptiste Caramiaux, Etienne Thoret
Please cite us if you use this script :)
All rights reserved

Implementation using a light implementation of the Matlab NLS toolbox by TODO

'''
import numpy as np
import math
from scipy import signal
import matplotlib.pylab as plt
import utils


def spectrum2scaletime(stft, num_channels, num_ch_oct, sr_time, nfft_rate,
                       nfft_scale, KIND):
    '''
    spectrum2scaletime
    '''
    lgt_time = stft.shape[0]
    mod_scale = np.zeros((lgt_time, nfft_scale), dtype=complex)
    phase_scale = np.zeros((lgt_time, nfft_scale))
    # perform a FFT for each time slice
    for i in range(lgt_time):
        mod_scale[i, :] = np.fft.fft(stft[i, :], nfft_scale)
        phase_scale[i, :] = utils.angle(mod_scale[i, :])
    mod_scale = np.abs(mod_scale)  # modulus of the fft
    scales = np.linspace(0, nfft_scale + 1, num_ch_oct)
    times = np.linspace(0, mod_scale.shape[1] + 1, int(lgt_time / sr_time))
    return mod_scale, phase_scale, times, scales


def scaletime2scalerate(mod_scale, num_channels, num_ch_oct, sr_time,
                        nfft_rate, nfft_scale, KIND):
    '''
    scaletime2scalerate
    '''
    lgt_scale = mod_scale.shape[1]
    scale_rate = np.zeros((nfft_rate, lgt_scale), dtype=complex)
    phase_scale_rate = np.zeros((nfft_rate, lgt_scale))
    for i in range(lgt_scale):
        scale_rate[:, i] = np.fft.fft(mod_scale[:, i], nfft_rate)
        phase_scale_rate[:, i] = utils.angle(scale_rate[:, i])
    scale_rate = np.abs(scale_rate)
    rates = np.linspace(0, nfft_rate + 1, sr_time)
    scales = np.linspace(0, nfft_scale + 1, num_ch_oct)
    return scale_rate, phase_scale_rate, rates, scales


def scalerate2cortical(stft, scaleRate, phase_scale_rate, scales, rates,
                       num_channels, num_ch_oct, sr_time, nfft_rate,
                       nfft_scale, KIND):
    LgtRateVector = len(rates)
    LgtScaleVector = len(scales)  # length scale vector
    LgtFreq = stft.shape[1]
    LgtTime = stft.shape[0]
    # plt.imshow(np.abs(stft))
    # plt.show()
    # plt.imshow(np.abs(scaleRate))
    # plt.show()

    cortical_rep = np.zeros(
        (LgtTime, LgtFreq, LgtScaleVector, LgtRateVector), dtype=complex)
    for j in range(LgtRateVector):
        fc_rate = rates[j]
        t = np.arange(nfft_rate / 2) / sr_time * abs(fc_rate)
        h = np.sin(2 * math.pi * t) * np.power(t, 2) * np.exp(
            -3.5 * t) * abs(fc_rate)
        h = h - np.mean(h)
        STRF_rate0 = np.fft.fft(h, nfft_rate)
        A = utils.angle(STRF_rate0[:nfft_rate // 2])
        A[0] = 0.0  # instead of pi
        STRF_rate = np.absolute(STRF_rate0[:nfft_rate // 2])
        STRF_rate = STRF_rate / np.max(STRF_rate)
        STRF_rate = STRF_rate * np.exp(1j * A)
        # rate filtering modification
        # STRF_rate                = [STRF_rate(1:nfft_rate/2); zeros(1,nfft_rate/2)']
        STRF_rate.resize((nfft_rate, ))
        STRF_rate[nfft_rate // 2] = np.absolute(STRF_rate[nfft_rate // 2 + 1])

        if (fc_rate < 0):
            STRF_rate[1:nfft_rate] = np.matrix.conjugate(
                np.flipud(STRF_rate[1:nfft_rate]))

        z1 = np.zeros((nfft_rate, nfft_scale // 2), dtype=complex)
        for m in range(nfft_scale // 2):
            z1[:, m] = STRF_rate * scaleRate[:, m] * np.exp(
                1j * phase_scale_rate[:, m])
        # z1.resize((nfft_rate,nfft_rate))
        for i in range(nfft_scale // 2):
            z1[:, i] = np.fft.ifft(z1[:, i])
        # print(z1[10,:])
        
        for i in range(LgtScaleVector):
            fc_scale = scales[i]

            R1 = np.arange(nfft_scale / 2) / (
                nfft_scale / 2) * num_ch_oct / 2 / abs(fc_scale)
            if KIND == 1:
                C1 = 1 / 2 / .3 / .3
                STRF_scale = np.exp(-C1 * np.power(R1 - 1, 2)) + np.exp(
                    -C1 * np.power(R1 + 1, 2))
            elif KIND == 2:
                R1 = np.power(R1, 2)
                STRF_scale = R1 * np.exp(1 - R1)
            z = np.zeros((LgtTime, nfft_scale // 2), dtype=complex)
           
            for n in range(LgtTime):
                temp = np.fft.ifft(STRF_scale * z1[n, :], nfft_scale)
                z[n, :] = temp[:nfft_scale // 2]            
            cortical_rep[:, :, i, j] = z[:LgtTime, :LgtFreq]
    #strf_avg = np.mean(cortical_rep, axis=(0, 1))
        
    return cortical_rep


#### NLS lite


def waveform2auditoryspectrogram(x_, frame_length, time_constant,
                                 compression_factor, octave_shift, filt, VERB):
    '''
    Wav2Aud form NSL toolbox
    @url http://www.isr.umd.edu/Labs/NSL/Software.htm
    '''

    # if (filt == 'k'):
    #     raise ValueError('Please use wav2aud_fir function for FIR filtering!')

    # if (filt == 'p_o'):
    #     COCHBA = np.genfromtxt('COCHBA_aud24_old.txt', dtype=str)
    # else:
    #     COCHBA = np.genfromtxt('COCHBA_aud24.txt', dtype=str)
    # # convert str to complex (may be a better way...)
    # COCHBA = np.asarray(
    #     [[complex(i.replace('i', 'j')) for i in COCHBA[row, :]]
    #      for row in range(len(COCHBA))])
    COCHBA = utils.COCHBA

    L, M = COCHBA.shape[0], COCHBA.shape[1]  # p_max = L - 2;
    L_x = len(x_)  # length of input

    # octave shift, nonlinear factor, frame length, leaky integration
    shft = octave_shift  #paras[3]  # octave shift
    fac = compression_factor  #paras[2]  # nonlinear factor
    L_frm = round(frame_length * 2**(4 + shft))  # frame length (points)

    alph = math.exp(-1 / (time_constant * 2**
                          (4 + shft))) if time_constant else 0

    # hair cell time constant in ms
    haircell_tc = 0.5
    beta = math.exp(-1 / (haircell_tc * 2**(4 + shft)))

    # get data, allocate memory for ouput
    N = math.ceil(L_x / L_frm)
    x = x_.copy()
    x.resize((N * L_frm, 1))  # zero-padding
    v5 = np.zeros((N, M - 1))

    #% last channel (highest frequency)
    p = COCHBA[0, M - 1].real
    B = COCHBA[np.arange(int(p) + 1) + 1, M - 1].real
    A = COCHBA[np.arange(int(p) + 1) + 1, M - 1].imag
    y1 = signal.lfilter(B, A, x, axis=0)
    y2 = utils.sigmoid(y1, fac)
    if (fac != -2):
        y2 = signal.lfilter([1.0], [1.0, -beta], y2, axis=0)
    y2_h = y2
    # % All other channels
    for ch in range((M - 2), -1, -1):
        # ANALYSIS: cochlear filterbank
        p = COCHBA[0, ch].real
        B = COCHBA[np.arange(int(p) + 1) + 1, ch].real
        A = COCHBA[np.arange(int(p) + 1) + 1, ch].imag
        y1 = signal.lfilter(B, A, x, axis=0)
        y2 = utils.sigmoid(y1, fac)
        # hair cell membrane (low-pass <= 4 kHz) ---> y2 (ignored for linear)
        if (fac != -2): y2 = signal.lfilter([1.0], [1.0, -beta], y2, axis=0)

        y3 = y2 - y2_h
        y2_h = y2
        # half-wave rectifier ---> y4
        y4 = np.maximum(y3, 0)

        # temporal integration window ---> y5
        if alph:  # leaky integration
            y5 = signal.lfilter([1.0], [1.0, -alph], y4, axis=0)
            v5[:, ch] = y5[L_frm * np.arange(1, N + 1) - 1].reshape(
                -1, )
        else:  # short-term average
            if (L_frm == 1):
                v5[:, ch] = y4
            else:
                v5[:, ch] = np.mean(y4.reshape(L_frm, N), axis=0)

    return v5


def complexSpectrogram(waveform, windowSize, frameStep):
    # % Figure out the fftSize (twice the window size because we are doing
    # % circular convolution).  We'll place the windowed time-domain signal into
    # % the middle of the buffer (zeros before and after the signal in the array.)
    fftSize = 2 * windowSize
    fftB = math.floor(windowSize / 2)
    fftE = fftB + windowSize
    fftBuffer = np.zeros((fftSize))

    # r, c = waveform.shape[0], waveform.shape[1]
    r = len(waveform)
    # if r > c:
    #     waveform = np.tranpose(waveform)

    frameCount = math.floor((r - windowSize) / frameStep) + 1

    spectrogram__ = np.zeros((fftSize, frameCount))
    # % h = hamming(windowSize)';
    h = 0.54 - 0.46 * np.cos(2 * math.pi * np.arange(windowSize) /
                             (windowSize - 1))
    # % h = h * 0 + 1;              % Just for debugging, no window.

    # % Note: This code loads the waveform data (times hamming) into the center
    # % of the fftSize buffer.  Then uses fftshift to rearrange things so that
    # % the 0-time is Matlab sample 1.  This means that the center of the window
    # % defines 0 phase.  After ifft, zero time will be at the same place.
    for frameNumber in range(frameCount):
        waveB = frameNumber * frameStep
        waveE = waveB + windowSize
        fftBuffer = 0.0 * fftBuffer  # make sure the buffer is empty
        fftBuffer[fftB:fftE] = waveform[waveB:waveE] * h
        fftBuffer = np.fft.fftshift(fftBuffer)
        # % fftBuffer(fftE+1:end) = 0;
        # % transpose (without the conjugate) into a column vector.
        # print(np.fft(fftBuffer).shape)
        spectrogram__[:, frameNumber] = np.transpose(np.abs(np.fft.fft(fftBuffer)))
    return spectrogram__
    # end
    # pass
