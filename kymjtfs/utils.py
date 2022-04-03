import numpy as np, torch


def sinusoid(f0, duration, sr):
    t = np.arange(0, duration, 1/sr)
    return torch.sin(2 * np.pi * f0 * t)


def gaussian(M, std, sym=True):
    ''' Gaussian window converted from scipy.signal.gaussian
    '''
    if M < 1:
        return torch.array([])
    if M == 1:
        return torch.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w


def generate_am_chirp(f_c, f_m, gamma, bw=2, duration=4, sr=2**13):
    sigma0 = 0.1
    t = torch.arange(-duration/2, duration/2, 1/sr)
    chirp_phase = 2*np.pi*f_c / (gamma*np.log(2)) * (2 ** (gamma*t) - 1)
    carrier = torch.sin(chirp_phase)
    modulator = torch.sin(2 * np.pi * f_m * t)
    window_std = sigma0 * bw / gamma
    window = gaussian(duration*sr, std=window_std*sr)
    x = carrier * modulator * window
    return x

def make_directory(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            print("Failed to create directory %s" % dir_path)
