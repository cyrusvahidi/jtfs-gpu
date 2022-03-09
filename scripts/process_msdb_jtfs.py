import os, fire
import numpy as np
from tqdm import tqdm

from kymjtfs.cnn import MedleyDataModule

from kymatio.torch import TimeFrequencyScattering1D


def make_directory(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            print("Failed to create directory %s" % dir_path)


class JTFSExtractor():


def process_msdb_jtfs(output_dir,
                      data_dir='/import/c4dm-datasets/medley-solos-db/',
                      jtfs_kwargs={
                        'shape': 2**16, 
                        'J': 12, 
                        'Q': 16, 
                        'F': 4, 
                        'T': 2**11,
                        'out_3D': True
                        'average_fr': True,
                        'max_pad_factor': 1,
                        'max_pad_factor_fr': 1,
                      }):
    pass


def main():
    fire.Fire(process_msdb_jtfs)


if __name__ == "__main__":
    main()