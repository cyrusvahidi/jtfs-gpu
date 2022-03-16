import os, fire
import numpy as np, torch
from tqdm import tqdm

from kymjtfs.cnn import MedleyDataModule

from kymatio.torch import TimeFrequencyScatteringTorch1D as TimeFrequencyScattering1D


def make_directory(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            print("Failed to create directory %s" % dir_path)


def normalize_audio(audio: np.ndarray, eps: float = 1e-10):
    max_val = max(np.abs(audio).max(), eps)

    return audio / max_val


class JTFSExtractor():

    def __init__(self,
                 output_dir,
                 jtfs_kwargs,
                 data_module):
        super().__init__()

        self.output_dir = output_dir
        self.jtfs_kwargs = jtfs_kwargs
        self.data_module = data_module

        self.jtfs = TimeFrequencyScattering1D(**jtfs_kwargs).cuda()
        
        self.lambda2_train = []

    def run(self):

        loaders = [('train', self.data_module.train_ds),
                   ('val', self.data_module.val_ds),
                   ('test', self.data_module.test_ds)]
                
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
                    self.lambda2_train.append(torch.concat([s1, s2]))

                Sx = [s.cpu().numpy() for s in Sx]
                out_path = os.path.join(subset_dir, os.path.splitext(fname)[0])
                np.save(out_path + '_S1', Sx[0])
                np.save(out_path + '_S2', Sx[1])

    def stats(self):
        print(f'Computing Mean Stat ...')
        samples = torch.stack(self.lambda2_train)
        self.mu = samples.mean(dim=0)
        stats_path = os.path.join(self.output_dir, 'stats')
        make_directory(stats_path)
        np.save(os.path.join(stats_path, 'mu'), self.mu.cpu().numpy())


def process_msdb_jtfs(output_dir='/import/c4dm-datasets/medley-solos-db/jtfs',
                      data_dir='/import/c4dm-datasets/medley-solos-db/',
                      jtfs_kwargs={
                        'shape': 2**16, 
                        'J': 12, 
                        'Q': 16, 
                        'F': 4, 
                        'T': 2**11,
                        'out_3D': True,
                        'average_fr': True,
                        'max_pad_factor': 1,
                        'max_pad_factor_fr': 1,
                      }):
    """ Script to save Medley-Solos-DB time-frequency scattering coefficients and stats
        to disk
    Args:
        output_dir: the output directory to save the numpy array features
        data_dir: source data directory for medley-solos-db download 
        jtfs_kwargs: kymatio jtfs frontend hyperparameters
    """
    make_directory(output_dir)
    data_module = MedleyDataModule(data_dir, batch_size=32)
    data_module.setup()

    extractor = JTFSExtractor(output_dir, jtfs_kwargs, data_module)
    extractor.run()
    extractor.stats()


def main():
    fire.Fire(process_msdb_jtfs)


if __name__ == "__main__":
    main()