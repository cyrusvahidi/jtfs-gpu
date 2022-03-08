import os, pandas as pd, numpy as np, torch, torchmetrics

from typing import Optional, Union

from torch.nn import functional as F
from torch import nn

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
import mirdata.datasets.medley_solos_db as msdb
from efficientnet_pytorch.model import EfficientNet

from kymatio.torch import TimeFrequencyScatteringTorch1D as TimeFrequencyScattering1D

from kymjtfs.batch_norm import ScatteringBatchNorm

from joblib import Memory


class MedleySolosClassifier(LightningModule):
    def __init__(self, 
                 in_shape = 2**16, 
                 J = 12, 
                 Q = 16, 
                 F = 4, 
                 T = 2**11, 
                 lr=5e-4, 
                 average='macro'):
        super().__init__()

        self.in_shape = in_shape
        self.J = J
        self.Q = Q
        self.F = F
        self.T = T
        
        self.lr = lr
        
        self.s1_conv1 = nn.Sequential(
            Unsqueeze(1),
            nn.Conv2d(1, 4, kernel_size=(16, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(4, 1), padding=(2, 0))
        )
        
        self.setup_jtfs()
        
        self.conv_net = EfficientNet.from_name('efficientnet-b0',
                                               in_channels=self.jtfs_channels,
                                               include_top = True,
                                               num_classes = 8)
        
        self.acc_metric = torchmetrics.Accuracy(num_classes=8, average=average)

    def setup_jtfs(self):
        self.jtfs = TimeFrequencyScattering1D(
            shape=(self.in_shape, ),
            T=self.T,
            Q=self.Q,
            J=self.J,
            F=self.F,
            average_fr=True,
            max_pad_factor=1, 
            max_pad_factor_fr=1,
            out_3D=True,)
        
        n_channels = self._get_jtfs_out_dim()
        
        self.jtfs_dim = self._get_jtfs_out_dim()
        self.jtfs_channels = self.jtfs_dim[0]
        self.jtfs_bn = ScatteringBatchNorm(self.jtfs_dim)
        
    def forward(self, x):
        Sx = x
        
        s1, s2 = Sx[0].squeeze(1), Sx[1].squeeze(1)
        s1_conv = self.s1_conv1(s1)
        s1_conv = F.pad(s1_conv, 
                   (0, 0, s2.shape[-2] - s1_conv.shape[-2], 0))
        
        sx = torch.cat([s1_conv, s2], dim=1)[:, :, :32, :]
        sx = torch.log1p(self.jtfs_bn(sx))
        y = self.conv_net(sx)
        y = F.log_softmax(y, dim=1)
        return y
    
        
    def step(self, batch, fold):
        Sx, y = batch
        logits = self(Sx)

        loss, acc = F.nll_loss(logits, y), self.acc_metric(logits, y)
        
        self.log(f'{fold}/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{fold}/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return {f'loss': loss, f'{fold}/acc': acc}
    
    def log_metrics(self, outputs, fold):
        keys = list(outputs[0].keys())
        for k in keys:
            metric = torch.stack([x[k] for x in outputs]).mean()
            self.log(f'{fold}/{k}', metric)
        
    def training_step(self, batch, batch_idx):
        return self.step(batch, fold='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, fold='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, fold='test')
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
        return [opt], [scheduler]
    
    def _get_jtfs_out_dim(self):
        dummy_in = torch.randn(self.in_shape).cuda()
        sx = self.jtfs(dummy_in)
        s1 = self.s1_conv1(sx[0])
        s1 = F.pad(s1, (0, 0, sx[1].shape[-2] - s1.shape[-2], 0))
        S = torch.cat([s1, sx[1]], dim=1)[:, :, :32, :]
        out_dim = S.shape[1:3]
        return out_dim
    

class Unsqueeze(nn.Module):
    def __init__(self, dim=1):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class MedleySolosDB(Dataset):
    def __init__(self, 
                 jtfs,
                 data_dir='/import/c4dm-datasets/medley-solos-db/', 
                 subset='training'):
        super().__init__()
        
        self.msdb = msdb.Dataset(data_dir)
        self.audio_dir = os.path.join(data_dir, 'audio')
        self.csv_dir = os.path.join(data_dir, 'annotation')
        self.subset = subset

        self.jtfs = jtfs

        df = pd.read_csv(os.path.join(self.csv_dir, 'Medley-solos-DB_metadata.csv'))
        self.df = df.loc[df['subset'] == subset]
        self.df.reset_index(inplace = True)

        # cachedir = '/import/c4dm-04/cv'
        # self.memory = Memory(cachedir, verbose=0)

        
    def build_audio_fname(self, df_item):
        uuid = df_item['uuid4']
        instr_id = df_item['instrument_id']
        subset = df_item['subset']
        return f'Medley-solos-DB_{subset}-{instr_id}_{uuid}.wav'

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        audio_fname = self.build_audio_fname(item)
        audio, _ = msdb.load_audio(os.path.join(self.audio_dir, audio_fname))

        # load_jtfs = self.memory.cache(_load_jtfs)
        # Sx = load_jtfs(self.jtfs, audio)
        Sx = self.jtfs(audio)
        y = int(item['instrument_id'])
        
        return Sx, y

    def __len__(self):
        return len(self.df)


def _load_jtfs(jtfs, audio):
    return jtfs(audio)


class MedleyDataModule(pl.LightningDataModule):
    def __init__(self, 
                 jtfs,
                 data_dir: str = '/import/c4dm-datasets/medley-solos-db/', 
                 batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.jtfs = jtfs

    def setup(self, stage: Optional[str] = None):
        self.train_ds = MedleySolosDB(self.jtfs, self.data_dir, subset='training')
        self.val_ds = MedleySolosDB(self.jtfs, self.data_dir, subset='validation')
        self.test_ds = MedleySolosDB(self.jtfs, self.data_dir, subset='test')

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, drop_last=True)
