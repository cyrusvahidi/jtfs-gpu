import os 
import fire, tqdm
import numpy as np, matplotlib.pyplot as plt, scipy
import librosa, librosa.feature, librosa.display

from sklearn.manifold import Isomap

def sinusoid(f0, duration, sr):
    t = np.arange(0, duration, 1/sr)
    return np.sin(2 * np.pi * f0 * t)


def generate(f_c, f_m, gamma, bw=2, duration=2, sr=2**14):
    sigma0 = 0.1
    t = np.arange(-duration/2, duration/2, 1/sr)
    chirp_phase = 2*np.pi*f_c / (gamma*np.log(2)) * (2 ** (gamma*t) - 1)
    carrier = np.sin(chirp_phase)
    modulator = np.sin(2 * np.pi * f_m * t)
    window_std = sigma0 * bw / gamma
    window = scipy.signal.gaussian(duration*sr, std=window_std*sr)
    x = carrier * modulator * window
    return x


def generate_audio(f0s, fms, gammas, duration, sr):
    audio = np.zeros((len(f0s), len(fms), len(gammas), duration * sr))
    cmap = np.zeros((3, len(f0s) * len(fms) * len(gammas)))
    c = 0

    print('Generating Audio ...')
    for i, f0 in tqdm.tqdm(enumerate(f0s)):
        for j, fm in enumerate(fms):
            for k, gamma in enumerate(gammas):
                audio[i, j, k, :] = generate(f0, fm, gamma, sr=sr, duration=duration)
                audio[i, j, k, :] = audio[i, j, k, :] / np.linalg.norm(audio[i, j, k, :])
                cmap[0, c], cmap[1, c], cmap[2, c] = f0, fm, gamma
                c += 1
    return audio, cmap


def extract_mfcc(audio, f0s, fms, gammas, sr, n_mfcc = 20):
    mfcc = np.zeros((len(f0s), len(fms), len(gammas), n_mfcc))

    print('Extracting MFCCs ...')
    for i, f0 in tqdm.tqdm(enumerate(f0s)):
        for j, fm in enumerate(fms):
            for k, gamma in enumerate(gammas):
                mfcc[i, j, k,:] = np.mean(librosa.feature.mfcc(y=audio[i,j,k], sr=sr), axis=-1)
    
    return mfcc.reshape(-1, mfcc.shape[-1])


def extract_time_scattering(audio, f0s, fms, gammas, sr, **ts_kwargs):
    pass


def extract_jtfs(audio, f0s, fms, gammas, sr, **jtfs_kwargs):
    pass


def extract_openl3(audio, f0s, fms, gammas, sr, **ol3_kwargs):
    pass


def extract_strf(audio, f0s, fms, gammas, sr, **strf_kwargs):
    pass


def plot_isomap(Y, cmap, out_dir):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.scatter3D(Y[:, 0], Y[:, 1], Y[:, 2], c=cmap[0], cmap='bwr');
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # f modulator
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.scatter3D(Y[:, 0], Y[:, 1], Y[:, 2], c=cmap[1], cmap='bwr');
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # chirp rate
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.scatter3D(Y[:, 0], Y[:, 1], Y[:, 2], c=cmap[2], cmap='bwr');
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)


    plt.savefig(os.path.join(out_dir, 'isomap.png'))


def run_isomaps(X, cmap, out_dir):

    Y = {}
    ratios = {}
    models = {}

    for feat in X.keys():
        feat_dir = os.path.join(out_dir, feat)

        os.makedirs(feat_dir, exist_ok=True)
        models[feat] = Isomap(n_components=3, n_neighbors=40)
        Y[feat] = models[feat].fit_transform(X[feat])

        plot_isomap(Y[feat], cmap, feat_dir)

        knn = models[feat].nbrs_.kneighbors()
        ratios[feat] = np.vstack([
            np.exp(np.mean(np.log(cmap[:, knn[1][i, :]]), axis=1)) / cmap[:, i]
            for i in range(X[feat].shape[0])
        ])



def run_isomap(
    n_steps = 16,
    f0_min = 512, 
    f0_max = 1024,
    fm_min = 4,
    fm_max = 16,
    gamma_min = 0.5,
    gamma_max = 4,
    bw = 2, 
    duration = 4, 
    sr = 2**13,
    out_dir = '/img'):


    out_dir = os.getcwd() + out_dir
    os.makedirs(out_dir, exist_ok=True)

    f0s = np.logspace(np.log10(f0_min), np.log10(f0_max), n_steps)
    fms = np.logspace(np.log10(fm_min), np.log10(fm_max), n_steps)
    gammas = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_steps)
    
    audio, cmap = generate_audio(f0s, fms, gammas, duration, sr)
    mfcc = extract_mfcc(audio, f0s, fms, gammas, sr)

    X = {"mfcc": mfcc}

    run_isomaps(X, cmap, out_dir)


def main():
  fire.Fire(run_isomap)


if __name__ == "__main__":
    main()