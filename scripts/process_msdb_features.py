import os, fire
import numpy as np, torch
from tqdm import tqdm
from pathlib import Path

from kymjtfs.cnn import MedleyDataModule
from kymjtfs.utils import make_abspath, fix_path_sep

from kymatio.torch import TimeFrequencyScattering1D, Scattering1D

from nnAudio.features import CQT
from torchaudio.transforms import AmplitudeToDB


def make_directory(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            print("Failed to create directory %s" % dir_path)


def normalize_audio(audio: np.ndarray, eps: float = 1e-10):
    return audio / (audio.std() + eps)


class Extractor():
    def __init__(self,
                 output_dir,
                 data_module):
        self.output_dir = make_abspath(output_dir)
        self.data_module = data_module

        self.lambda_train = []

    def get_loaders(self):
        loaders = [('training', self.data_module.train_ds),
                   ('validation', self.data_module.val_ds),
                   ('test', self.data_module.test_ds)]
        return loaders

    def stats(self):
        print('Computing Mean Stat ...')
        samples = torch.stack(self.lambda_train)
        self.mu = samples.mean(dim=0)
        stats_path = os.path.join(self.output_dir, 'stats')
        make_directory(stats_path)
        np.save(os.path.join(stats_path, 'mu'), self.mu.cpu().numpy())


class CQTExtractor(Extractor):

    def __init__(self,
                 output_dir,
                 data_module,
                 cqt_kwargs={
                    'sr': 44100,
                    'n_bins': 96,
                    'hop_length': 256,
                    'fmin': 32.7}):
        super().__init__(output_dir, data_module)
        self.output_dir = make_abspath(output_dir)
        self.data_module = data_module
        self.cqt_kwargs = cqt_kwargs

        self.cqt = CQT(**cqt_kwargs).cuda()
        self.a_to_db = AmplitudeToDB(stype = 'magnitude').cuda()

        self.samples = []


    def run(self):

        loaders = self.get_loaders()

        for subset, loader in loaders:
            subset_dir = os.path.join(self.output_dir, subset)
            make_directory(subset_dir)
            print(f'Extracting CQT for {subset} set ...')
            for idx, item in tqdm(enumerate(loader)):
                audio, _, fname = item
                audio = normalize_audio(audio)
                audio = torch.tensor(audio).cuda()
                Sx = self.a_to_db(self.cqt(audio))[0]
                out_path = os.path.join(subset_dir, os.path.splitext(fname)[0])
                np.save(out_path, Sx.cpu().numpy())

                self.samples.append(Sx.mean(dim=-1))

    def stats(self):
        print('Computing Mean Stat ...')
        samples = torch.stack(self.samples)
        self.mu = samples.mean(dim=0)
        self.std = samples.std(dim=0)
        stats_path = os.path.join(self.output_dir, 'stats')
        make_directory(stats_path)
        np.save(os.path.join(stats_path, 'mu'), self.mu.cpu().numpy())
        np.save(os.path.join(stats_path, 'std'), self.std.cpu().numpy())


class JTFSExtractor(Extractor):

    def __init__(self,
                 output_dir,
                 data_module,
                 jtfs_kwargs=None):
        super().__init__(output_dir, data_module)
        self.output_dir = make_abspath(output_dir)
        self.jtfs_kwargs = jtfs_kwargs
        self.data_module = data_module

        self.lambda2_train = []

        # build jtfs #########################################
        # included almost all args for forward-compatibility
        default_main = {
            'shape': 2**16,
            'J': (13, 13),
            'Q': (16, 1),
            'T': 2**11,

            'J_fr': 6,
            'Q_fr': 1,
            'F': 4,
            'average_fr': True,
            'out_3D': True,
            'max_pad_factor_fr': 3,
            'sampling_filters_fr': ('exclude', 'resample'),
        }
        default_extra = {
            'average': True,
            'pad_mode': 'reflect',
            'normalize': 'l1-energy',
            'oversampling': 0,
            'max_pad_factor': 2,

            'aligned': True,
            'analytic': True,
            'oversampling_fr': 0,
        }
        default = {**default_main, **default_extra}
        if jtfs_kwargs is None:
            jtfs_kwargs = default.copy()
        else:
            # fill what's unspecified
            for k, v in default.items():
                if k not in jtfs_kwargs:
                    jtfs_kwargs[k] = v
        self.jtfs = TimeFrequencyScattering1D(**jtfs_kwargs).cuda()

    def run(self):

        loaders = self.get_loaders()

        for subset, loader in loaders:
            subset_dir = os.path.join(self.output_dir, subset)
            make_directory(subset_dir)
            names = os.listdir(subset_dir)
            print(f'Extracting JTFS for {subset} set ...')
            for idx, item in tqdm(enumerate(loader)):
                if item is None:
                    continue
                audio, _, fname = item
                out_path = os.path.join(subset_dir, os.path.splitext(fname)[0])

                # if os.path.basename(path0) in os.listdir(subset_dir):
                #     print(end='.')
                #     continue

                audio = normalize_audio(audio)
                Sx = self.jtfs(audio)

                if self.jtfs_kwargs['out_3D']:
                    if subset == 'training':
                        # collect S1 and S2 integrated over time and lambda
                        s1, s2 = Sx[0][0], Sx[1][0]

                        # self.lambda_train.append(s1[1:])
                        # self.lambda2_train.append(s2)
                    Sx = [s.cpu().numpy() for s in Sx]
                    np.save(out_path + '_S1', Sx[0])
                    np.save(out_path + '_S2', Sx[1])
                else:
                    Sx = Sx[0]
                    if subset == 'training':
                        # collect S1 and S2 integrated over time and lambda
                        s_mu = Sx.mean(dim=-1)
                        # self.lambda_train.append(s_mu)
                    out_path = os.path.join(subset_dir, os.path.splitext(fname)[0])
                    np.save(out_path, Sx.cpu().numpy())

    def stats(self, version=None):
        print('Computing Mean Stat ...')
        if version is None:
            try:
                (mu_z_s1, mu_z_s2, std_z_s1, std_z_s2
                 ) = self.stats_on_gpu()
            except:
                try:
                    (mu_z_s1, mu_z_s2, std_z_s1, std_z_s2
                     ) = self.stats_on_cpu('vectorized')
                except:
                    (mu_z_s1, mu_z_s2, std_z_s1, std_z_s2
                     ) = self.stats_on_cpu('loop')
        else:
            fn = {'gpu': self.stats_on_gpu,
                  'vectorized': lambda: self.stats_on_cpu('vectorized'),
                  'loop': lambda: self.stats_on_cpu('loop')}[version]
            mu_z_s1, mu_z_s2, std_z_s1, std_z_s2 = fn()

        def cpu(x):
            return (x if isinstance(x, np.ndarray) else
                    x.cpu().numpy())

        stats_path = os.path.join(self.output_dir, 'stats')
        make_directory(stats_path)
        def save_as_numpy(path, arr):
            np.save(os.path.join(stats_path, path), cpu(arr))

        save_as_numpy('mu_s1', self.mu_s1)
        save_as_numpy('mu_s2', self.mu_s2)
        save_as_numpy('mu_z_s1', mu_z_s1)
        save_as_numpy('mu_z_s2', mu_z_s2)
        save_as_numpy('std_z_s1', std_z_s1)
        save_as_numpy('std_z_s2', std_z_s2)

    def stats_on_gpu(self):
        samples_s1 = torch.stack(self.lambda_train)
        samples_s2 = torch.stack(self.lambda2_train)
        self.mu_s1 = samples_s1.mean(dim=-1).mean(dim=0)
        self.mu_s2 = samples_s2.mean(dim=-1).mean(dim=-1).mean(dim=0)

        s1_renorm = torch.log1p(samples_s1 /
                                (1e-1 * self.mu_s1[None, :, None] + 1e-8)
                                ).mean(dim=-1)
        s2_renorm = torch.log1p(samples_s2 /
                                (1e-1 * self.mu_s2[None, :, None, None] + 1e-8)
                                ).mean(dim=-1).mean(dim=-1)
        mu_z_s1, std_z_s1 = s1_renorm.mean(dim=0), s1_renorm.std(dim=0)
        mu_z_s2, std_z_s2 = s2_renorm.mean(dim=0), s2_renorm.std(dim=0)

        return mu_z_s1, mu_z_s2, std_z_s1, std_z_s2

    def stats_on_cpu(self, version):
        paths = [str(p) for p in Path(self.output_dir, 'training').iterdir()
                 if p.suffix == '.npy']
        paths_s1 = [p for p in paths if Path(p).stem.endswith('_S1')]
        paths_s2 = [p for p in paths if Path(p).stem.endswith('_S2')]
        paths_s1.sort(key=lambda p: os.path.getctime(p))
        paths_s2.sort(key=lambda p: os.path.getctime(p))

        if version == 'vectorized':
            lambda_train, lambda2_train = [], []
        else:
            S1_ref, S2_ref = np.load(paths_s1[0]), np.load(paths_s2[0])
            S1_ref = S1_ref[:, 1:]  # drop S0
            samples_s1, samples_s2 = np.zeros(S1_ref.shape), np.zeros(S2_ref.shape)
            lambda_train_lens  = np.zeros(S1_ref.ndim)
            lambda2_train_lens = np.zeros(S2_ref.ndim)

        for p1, p2 in zip(paths_s1, paths_s2):
            S1, S2 = np.load(p1), np.load(p2)
            S1 = S1[:, 1:]  # drop S0

            if version == 'vectorized':
                lambda_train.append(S1)
                lambda2_train.append(S2)
            else:
                lambda_train_lens  += np.array(S1.shape)
                lambda2_train_lens += np.array(S2.shape)
                samples_s1 += S1
                samples_s2 += S2

        if version == 'vectorized':
            samples_s1, samples_s2 = [], []
            samples_s1 = np.stack(lambda_train).squeeze(1)
            samples_s2 = np.stack(lambda2_train).squeeze(1)
            mu_s1 = samples_s1.mean(axis=-1).mean(axis=0)
            mu_s2 = samples_s2.mean(axis=-1).mean(axis=-1).mean(axis=0)
        else:
            mu_s1 = samples_s1.mean(axis=-1) / lambda_train_lens[0]
            mu_s2 = samples_s2.mean(axis=-1).mean(axis=-1) / lambda2_train_lens[0]
            mu_s1, mu_s2 = mu_s1.squeeze(0), mu_s2.squeeze(0)
        self.mu_s1, self.mu_s2 = mu_s1, mu_s2

        mu_div_s1 = (1e-1 * mu_s1[None, :, None] + 1e-8)
        mu_div_s2 = (1e-1 * mu_s2[None, :, None, None] + 1e-8)

        if version == 'vectorized':
            S1_renorm = np.log1p(samples_s1 / mu_div_s1).mean(axis=-1)
            S2_renorm = np.log1p(samples_s2 / mu_div_s2).mean(axis=-1
                                                              ).mean(axis=-1)
            mu_z_s1, std_z_s1 = S1_renorm.mean(axis=0), S1_renorm.std(axis=0)
            mu_z_s2, std_z_s2 = S2_renorm.mean(axis=0), S2_renorm.std(axis=0)
        else:
            S1_renorm = np.zeros(samples_s1.shape[:-1])
            S2_renorm = np.zeros(samples_s2.shape[:-2])

            for p1, p2 in zip(paths_s1, paths_s2):
                S1, S2 = np.load(p1), np.load(p2)
                S1 = S1[:, 1:]  # drop S0

                S1_renorm += np.log1p(S1 / mu_div_s1).mean(axis=-1)
                S2_renorm += np.log1p(S2 / mu_div_s2).mean(axis=-1).mean(axis=-1)

            mu_z_s1 = S1_renorm / lambda_train_lens[0]
            mu_z_s2 = S2_renorm / lambda2_train_lens[0]
            mu_z_s1, mu_z_s2 = mu_z_s1.squeeze(), mu_z_s2.squeeze()

        if version == 'loop':
            S1_renorm = np.zeros(samples_s1.shape[:-1])
            S2_renorm = np.zeros(samples_s2.shape[:-2])

            for p1, p2 in zip(paths_s1, paths_s2):
                S1, S2 = np.load(p1), np.load(p2)
                S1 = S1[:, 1:]  # drop S0

                S1_renorm += (np.log1p(S1 / mu_div_s1).mean(axis=-1)
                              - mu_z_s1)**2
                S2_renorm += (np.log1p(S2 / mu_div_s2).mean(axis=-1).mean(axis=-1)
                              - mu_z_s2)**2
            std_z_s1 = np.sqrt(S1_renorm / lambda_train_lens[0])
            std_z_s2 = np.sqrt(S2_renorm / lambda2_train_lens[0])
            std_z_s1, std_z_s2 = std_z_s1.squeeze(0), std_z_s2.squeeze(0)

        return mu_z_s1, mu_z_s2, std_z_s1, std_z_s2


class Scat1DExtractor(Extractor):

    def __init__(self,
                 output_dir,
                 data_module,
                 scat1d_kwargs={
                    'shape': 2**16,
                    'J': 13,
                    'T': 2**11,
                    'Q': 16,
                    'max_pad_factor': 3}):
        super().__init__(output_dir, data_module)
        self.output_dir = make_abspath(output_dir)
        self.data_module = data_module
        self.scat1d_kwargs = scat1d_kwargs

        self.scat1d = Scattering1D(**scat1d_kwargs).cuda()
        meta = self.scat1d.meta()
        order1 = np.where(meta['order'] == 1)
        order2 = np.where(meta['order'] == 2)
        self.idxs = np.concatenate([order1[0], order2[0]])

    def run(self):

        loaders = self.get_loaders()

        for subset, loader in loaders:
            subset_dir = os.path.join(self.output_dir, subset)
            make_directory(subset_dir)
            print(f'Extracting Scat1D for {subset} set ...')
            for idx, item in tqdm(enumerate(loader)):
                audio, _, fname = item
                audio = normalize_audio(audio)
                audio = torch.tensor(audio).cuda()
                Sx = self.scat1d(audio)[self.idxs]
                if subset == 'training':
                    # collect integrated over time
                    self.lambda_train.append(Sx.mean(dim=-1).cpu().numpy())

                out_path = os.path.join(subset_dir, os.path.splitext(fname)[0])
                np.save(out_path, Sx.cpu().numpy())


data_dir='import/c4dm-datasets/medley-solos-db/'
feature='jtfs'
out_dir_id=''
# """ Script to save Medley-Solos-DB time-frequency scattering coefficients
#     and stats to disk
# Args:
#     data_dir: source data directory for medley-solos-db download
#     feature: ['jtfs', 'scat1d', 'cqt']
#     output_dir_id: optional identifier to append to the output dir name
# """
output_dir = os.path.join(make_abspath(data_dir),
                          fix_path_sep(feature + out_dir_id))
make_directory(output_dir)
data_module = MedleyDataModule(data_dir, batch_size=32, feature='',
                               out_dir_to_skip=output_dir)
data_module.setup()

if feature == 'jtfs':
    extractor = JTFSExtractor(output_dir, data_module)
elif feature == 'scat1d':
    extractor = Scat1DExtractor(output_dir, data_module)
elif feature == 'cqt':
    extractor = CQTExtractor(output_dir, data_module)

# extractor.run()
# extractor.stats()


# def main():
#     fire.Fire(process_msdb_jtfs)


# if __name__ == "__main__":
#     main()
