import os, fire
import numpy as np, torch
from tqdm import tqdm

from kymjtfs.cnn import MedleyDataModule

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
    max_val = max(np.abs(audio).max(), eps)

    return audio / max_val


class Extractor():
    def __init__(self,
                output_dir,
                data_module):
        self.output_dir = output_dir 
        self.data_module = data_module

        self.lambda_train = []

    def get_loaders(self):
        loaders = [('training', self.data_module.train_ds),
                   ('validation', self.data_module.val_ds),
                   ('test', self.data_module.test_ds)]
        return loaders

    def stats(self):
        print(f'Computing Mean Stat ...')
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
        self.output_dir = output_dir
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
        print(f'Computing Mean Stat ...')
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
                 jtfs_kwargs={
                    'shape': 2**16, 
                    'J': 8, 
                    'Q': 16, 
                    'F': 4, 
                    'T': 2**11,
                    'out_3D': True,
                    'average_fr': True,
                    'max_pad_factor': 3,
                    'max_pad_factor_fr': 3}):
        super().__init__(output_dir, data_module)
        self.output_dir = output_dir
        self.jtfs_kwargs = jtfs_kwargs
        self.data_module = data_module

        self.jtfs = TimeFrequencyScattering1D(**jtfs_kwargs).cuda()

        self.lambda2_train = []
    
    def run(self):

        loaders = self.get_loaders()
                
        for subset, loader in loaders:
            subset_dir = os.path.join(self.output_dir, subset)
            make_directory(subset_dir)
            print(f'Extracting JTFS for {subset} set ...')
            for idx, item in tqdm(enumerate(loader)): 
                audio, _, fname = item
                audio = normalize_audio(audio)
                Sx = self.jtfs(audio)

                if self.jtfs_kwargs['out_3D']:
                    if subset == 'training':
                        # collect S1 and S2 integrated over time and lambda
                        s1, s2 = Sx[0][0], Sx[1][0]
                        # self.lambda_train.append(torch.concat([s1, s2]))
                        self.lambda_train.append(s1[1:])
                        self.lambda2_train.append(s2)
                    Sx = [s.cpu().numpy() for s in Sx]
                    out_path = os.path.join(subset_dir, os.path.splitext(fname)[0])
                    np.save(out_path + '_S1', Sx[0])
                    np.save(out_path + '_S2', Sx[1])
                else: 
                    Sx = Sx[0]
                    if subset == 'training':
                        # collect S1 and S2 integrated over time and lambda
                        s_mu = Sx.mean(dim=-1)
                        self.lambda_train.append(s_mu)
                    out_path = os.path.join(subset_dir, os.path.splitext(fname)[0])
                    np.save(out_path, Sx.cpu().numpy())
    
    def stats(self):
        print(f'Computing Mean Stat ...')
        samples_s1 = torch.stack(self.lambda_train)
        samples_s2 = torch.stack(self.lambda2_train)
        self.mu_s1 = samples_s1.mean(dim=-1).mean(dim=0)
        self.mu_s2 = samples_s2.mean(dim=-1).mean(dim=-1).mean(dim=0)

        s1_renorm = torch.log1p(samples_s1 / (1e-1 * self.mu_s1[None, :, None] + 1e-8)).mean(dim=-1)
        s2_renorm = torch.log1p(samples_s2 / (1e-1 * self.mu_s2[None, :, None, None] + 1e-8)).mean(dim=-1).mean(dim=-1)
        mu_z_s1, std_z_s1 = s1_renorm.mean(dim=0), s1_renorm.std(dim=0)
        mu_z_s2, std_z_s2 = s2_renorm.mean(dim=0), s2_renorm.std(dim=0)

        stats_path = os.path.join(self.output_dir, 'stats')
        make_directory(stats_path)
        np.save(os.path.join(stats_path, 'mu_s1'), self.mu_s1.cpu().numpy())
        np.save(os.path.join(stats_path, 'mu_s2'), self.mu_s2.cpu().numpy())
        np.save(os.path.join(stats_path, 'mu_z_s1'), mu_z_s1.cpu().numpy())
        np.save(os.path.join(stats_path, 'mu_z_s2'), mu_z_s2.cpu().numpy())
        np.save(os.path.join(stats_path, 'std_z_s1'), std_z_s1.cpu().numpy())
        np.save(os.path.join(stats_path, 'std_z_s2'), std_z_s2.cpu().numpy())


class Scat1DExtractor(Extractor):

    def __init__(self,
                 output_dir,
                 data_module,
                 scat1d_kwargs={
                    'shape': 2**16, 
                    'J': 8, 
                    'T': 2**11,
                    'Q': 16}):
        super().__init__(output_dir, data_module)
        self.output_dir = output_dir
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
                    self.lambda_train.append(Sx.mean(dim=-1))

                out_path = os.path.join(subset_dir, os.path.splitext(fname)[0])
                np.save(out_path, Sx.cpu().numpy())


def process_msdb_jtfs(data_dir='/import/c4dm-datasets/medley-solos-db/',
                      feature='jtfs',
                      out_dir_id='_test'):
    """ Script to save Medley-Solos-DB time-frequency scattering coefficients and stats
        to disk
    Args:
        output_dir: the output directory to save the numpy array features
        data_dir: source data directory for medley-solos-db download 
        feature: ['jtfs', 'scat1d', 'cqt']
    """
    output_dir = os.path.join(data_dir, feature + out_dir_id)
    make_directory(output_dir)
    data_module = MedleyDataModule(data_dir, batch_size=32, feature='')
    data_module.setup()
    
    if feature == 'jtfs':
        extractor = JTFSExtractor(output_dir, data_module)
    elif feature == 'scat1d':
        extractor = Scat1DExtractor(output_dir, data_module)
    elif feature == 'cqt':
        extractor = CQTExtractor(output_dir, data_module)
    extractor.run()
    extractor.stats()


def main():
    fire.Fire(process_msdb_jtfs)


if __name__ == "__main__":
    main()