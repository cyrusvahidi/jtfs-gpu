import os, fire
import numpy as np, torch
from tqdm import tqdm

from kymjtfs.cnn import MedleyDataModule

from kymatio.torch import TimeFrequencyScatteringTorch1D as TimeFrequencyScattering1D, Scattering1D


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
        loaders = [('train', self.data_module.train_ds),
                ('val', self.data_module.val_ds),
                ('test', self.data_module.test_ds)]
        return loaders

    def stats(self):
        print(f'Computing Mean Stat ...')
        samples = torch.stack(self.lambda_train)
        self.mu = samples.mean(dim=0)
        stats_path = os.path.join(self.output_dir, 'stats')
        make_directory(stats_path)
        np.save(os.path.join(stats_path, 'mu'), self.mu.cpu().numpy())


class JTFSExtractor(Extractor):

    def __init__(self,
                 output_dir,
                 data_module,
                 jtfs_kwargs={
                    'shape': 2**16, 
                    'J': 12, 
                    'Q': 16, 
                    'F': 4, 
                    'T': 2**11,
                    'out_3D': True,
                    'average_fr': True,
                    'max_pad_factor': 1,
                    'max_pad_factor_fr': 1}):
        super().__init__(output_dir, data_module)
        self.output_dir = output_dir
        self.jtfs_kwargs = jtfs_kwargs
        self.data_module = data_module

        self.jtfs = TimeFrequencyScattering1D(**jtfs_kwargs).cuda()
    
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

                if subset == 'train':
                    # collect S1 and S2 integrated over time and lambda
                    s1, s2 = Sx[0].mean(dim=-1).mean(dim=-1), Sx[1].mean(dim=-1).mean(dim=-1)[0, :]
                    self.lambda_train.append(torch.concat([s1, s2]))

                Sx = [s.cpu().numpy() for s in Sx]
                out_path = os.path.join(subset_dir, os.path.splitext(fname)[0])
                np.save(out_path + '_S1', Sx[0])
                np.save(out_path + '_S2', Sx[1])


class Scat1DExtractor(Extractor):

    def __init__(self,
                 output_dir,
                 data_module,
                 scat1d_kwargs={
                    'shape': 2**16, 
                    'J': 12, 
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
                if subset == 'train':
                    # collect integrated over time
                    self.lambda_train.append(Sx.mean(dim=-1))

                out_path = os.path.join(subset_dir, os.path.splitext(fname)[0])
                np.save(out_path, Sx)


def process_msdb_jtfs(data_dir='/import/c4dm-datasets/medley-solos-db/',
                      feature='jtfs'):
    """ Script to save Medley-Solos-DB time-frequency scattering coefficients and stats
        to disk
    Args:
        output_dir: the output directory to save the numpy array features
        data_dir: source data directory for medley-solos-db download 
        feature: ['jtfs', 'scat1d']
    """
    output_dir = os.path.join(data_dir, feature)
    make_directory(output_dir)
    data_module = MedleyDataModule(data_dir, batch_size=32)
    data_module.setup()
    
    if feature == 'jtfs':
        extractor = JTFSExtractor(output_dir, data_module)
    elif feature == 'scat1d':
        extractor = Scat1DExtractor(output_dir, data_module)
    extractor.run()
    extractor.stats()


def main():
    fire.Fire(process_msdb_jtfs)


if __name__ == "__main__":
    main()