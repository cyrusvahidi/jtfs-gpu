import numpy as np


def generate_harmonic_signal(T, num_intervals=4, gamma=0.9, random_state=42, n_harmonics=5):
    """
    Generates a harmonic signal, which is made of piecewise constant notes
    (of random fundamental frequency), with half overlap
    """
    rng = np.random.RandomState(random_state)
    num_notes = 2 * (num_intervals - 1) + 1
    support = T // num_intervals
    half_support = support // 2

    base_freq = 0.1 * rng.rand(num_notes) + 0.05
    phase = 2 * np.pi * rng.rand(num_notes)
    window = np.hanning(support)
    x = np.zeros(T, dtype='float32')
    t = np.arange(0, support)
    u = 2 * np.pi * t
    for i in range(num_notes):
        ind_start = i * half_support
        note = np.zeros(support)
        for k in range(n_harmonics):
            note += (np.power(gamma, k) *
                     np.cos(u * (k + 1) * base_freq[i] + phase[i]))
        x[ind_start:ind_start + support] += note * window

    return x


def chirp(T, f0, f1 = None, fs=1024, method='exp'):
    assert f0 >= 0 and f1 >= 0, "start (f0) and end (f1) frequencies must be positive"
    f1 = fs // 2 if not f1 else f1
    t = np.arange(0, T, 1)
    if method == 'lin':
        c = (f1 - f0) / T
        phi = np.cumsum(2 * np.pi * (f0 + c * t) * (1 / fs))
    if method == 'exp':
        k = (f1 / f0) ** (1 / T)
        phi = np.cumsum(2 * np.pi * (f0 * k ** t) * (1 / fs))
    return np.cos(phi)


def exp_chirp(T=1024, f0=10, rate=2):
    t = np.linspace(0, 1, T)
#     phi = 2 * np.pi * ((f0 * (np.exp(2 * t) - 1)) / rate)
    phi = 2 * np.pi * (f0 * (2 ** (rate * t) - 1)) / (rate * np.log(2))
    return np.cos(phi)
    