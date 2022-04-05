from typing import Optional, Union

import os, pandas as pd, numpy as np, torch, librosa, gin
import pytorch_lightning as pl
import mirdata.datasets.medley_solos_db as msdb
import torchvision.models as models

from torch.nn import functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import AmplitudeToDB
from torchmetrics import Accuracy, ClasswiseWrapper
from pytorch_lightning.core.lightning import LightningModule
from nnAudio.features import CQT
from kymatio.torch import TimeFrequencyScattering1D

from kymjtfs.batch_norm import ScatteringBatchNorm


@gin.configurable
class MedleySolosClassifier(LightningModule):
    def __init__(self, 
                 c = 1e-1,
                 jtfs_kwargs = {'shape': (2**16, ),
                                'J': 12,
                                'Q': 16,
                                'F': 4,
                                'T': 2**11},
                 lr=1e-3,
                 average='weighted', 
                 stats_dir='/import/c4dm-datasets/medley-solos-db/',
                 classes=['clarinet', 
                          'distorted electric guitar',
                          'female singer',
                          'flute',
                          'piano',
                          'tenor saxophone',
                          'trumpet',
                          'violin'],
                 csv='/import/c4dm-datasets/medley-solos-db/annotation/Medley-solos-DB_metadata.csv',
                 feature='jtfs',
                 learn_adalog = True):
        super().__init__()

        self.jtfs_kwargs = jtfs_kwargs
        self.lr = lr
        self.feature = feature.split('_')[0]
        self.feature_spec = feature.split('_')[1:]
        self.learn_adalog = learn_adalog

        self.acc_metric = Accuracy(num_classes=len(classes), average=average)
        self.acc_metric_macro = Accuracy(num_classes=len(classes), average='macro')
        self.classwise_acc = ClasswiseWrapper(Accuracy(num_classes=len(classes), average=None), 
                                                       labels=classes)
        self.loss = nn.CrossEntropyLoss(weight=self.get_class_weight(csv))
        
        self.is_2d_conv = self.feature == 'cqt' or (self.feature == 'jtfs' and '3D' in self.feature_spec)

        if self.feature == 'jtfs' or self.feature == 'scat1d':
            stats_dir = os.path.join(stats_dir, feature)
            self.mu = torch.tensor(np.load(os.path.join(stats_dir, 'stats/mu.npy')))
            
            if self.feature == 'jtfs' and '3D' in self.feature_spec:
                s1_channels = 4
                self.s1_conv1 = nn.Sequential(
                    nn.Conv2d(1, s1_channels, kernel_size=(16, 1)),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=(4, 1), padding=(2, 0))
                )
                self.n_channels = len(self.mu) + (s1_channels - 1)
            elif self.feature == 'scat1d' or (self.feature == 'jtfs' and '2D' in self.feature_spec):
                self.n_channels = len(self.mu)

            self.c = c 
            if self.learn_adalog:
                self.register_parameter('eps', nn.Parameter(torch.randn(len(self.mu))))
        elif feature == 'cqt':
            self.n_channels = 1
            # self.cqt = CQT(sr=44100, n_bins=96, hop_length=256, fmin=32.7)
            # self.a_to_db = AmplitudeToDB(stype = 'magnitude')
        
        self.bn = nn.BatchNorm2d(self.n_channels) if self.is_2d_conv else nn.BatchNorm1d(self.n_channels)

        self.setup_cnn(len(classes))                                                 
         
    def setup_cnn(self, num_classes):
        if self.is_2d_conv:
            # self.conv_net = LeNet(num_classes, self.n_channels)
            self.conv_net = models.efficientnet_b0()
            # modify input channels 
            self.conv_net.features[0][0] = nn.Conv2d(self.n_channels, 
                                                    32, 
                                                    kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), 
                                                    bias=False)
            self.conv_net.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)    
        else:
            # 1d convnet
            # self.conv_net = LeNet1D(num_classes, self.n_channels)
            self.conv_net = EfficientNet1d(self.n_channels, num_classes)

    def setup_jtfs(self):
        self.jtfs = TimeFrequencyScattering1D(
            **jtfs_kwargs,
            average_fr=True,
            max_pad_factor=1, 
            max_pad_factor_fr=1,
            out_3D=True,)
        
        n_channels = self._get_jtfs_out_dim()
        
        self.jtfs_dim = self._get_jtfs_out_dim()
        self.jtfs_channels = self.jtfs_dim[0]

    def get_class_weight(self, csv):
        df = pd.read_csv(csv)
        supports = list(df['instrument'].value_counts(sort=False))
        weight = [max(supports) / s for s in supports]
        return torch.tensor(weight)
        
    def forward(self, x):
        if self.feature == 'jtfs' and self.is_2d_conv:
            Sx = x
            s1, s2 = Sx[0].squeeze(1), Sx[1].squeeze(1)

            # apply AdaLog
            c = self.get_c().type_as(s1)
            if c.shape[0] == 1:
                c1, c2 = c, c
            else:
                c1, c2 = c[None, :1, None], c[None, 1:, None, None]

            s1 = s1 / (c1 * self.mu[None, :1, None].type_as(s1))
            s2 = s2 / (c2 * self.mu[None, 1:, None, None].type_as(s2))

            # s1 learnable frequential filter
            s1_conv = self.s1_conv1(s1.unsqueeze(1))
            s1_conv = F.pad(s1_conv, (0, 0, s2.shape[-2] - s1_conv.shape[-2], 0))
            sx = torch.cat([s1_conv, s2], dim=1)
            # log1p and batch norm
            sx = torch.log1p(sx)
            sx = self.bn(sx)
        elif self.feature == 'scat1d' or (self.feature == 'jtfs' and not self.is_2d_conv):
            Sx = x
            c = self.get_c().type_as(Sx)
            sx = torch.log1p(Sx / (c[None, :, None] * self.mu[None, :, None].type_as(Sx) + 1e-8))
            sx = self.bn(sx)
        elif self.feature == 'cqt':
            # X = self.a_to_db(self.cqt(x))
            sx = F.avg_pool2d(x, kernel_size=(3, 8))
            sx = sx.unsqueeze(1)
            sx = self.bn(sx)

        # conv net
        y = self.conv_net(sx)
        
        return y
 
    def step(self, batch, fold):
        Sx, y = batch
        logits = self(Sx)

        loss, acc = self.loss(logits, y), self.acc_metric(logits, y)
        class_acc = self.classwise_acc(logits, y)
        class_acc = {k: float(v.detach()) for k, v in class_acc.items()}
        
        self.log(f'{fold}/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{fold}/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        if fold == 'test':
            # macro_avg = self.acc_metric_macro(logits, y)
            # self.log(f'{fold}/avg_macro', macro_avg, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{fold}/classwise', class_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return {f'loss': loss, f'{fold}/acc': acc, f'{fold}/classwise': class_acc}
    
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 
                                                               factor=0.5, 
                                                               patience=4, 
                                                               mode='min')
        # scheduler = torch.optim.lr_scheduler.CyclicLR(opt, 
        #                                               base_lr=self.lr, 
        #                                               max_lr=self.lr * 4, 
        #                                               step_size_up=1024, 
        #                                               mode='triangular', 
        #                                               gamma=1.0, 
        #                                               cycle_momentum=False)
        # return {'optimizer': opt, 'lr_scheduler': scheduler}
        # return {'optimizer': opt}
        return {'optimizer': opt, 'lr_scheduler': scheduler, 'monitor':  'val/loss'}

    def get_c(self):
        c = (self.c * torch.exp(torch.tanh(self.eps))) if self.learn_adalog else torch.tensor([self.c])
        return c
    
    def _get_jtfs_out_dim(self):
        dummy_in = torch.randn(self.in_shape).cuda()
        sx = self.jtfs(dummy_in)
        s1 = self.s1_conv1(sx[0])
        s1 = F.pad(s1, (0, 0, sx[1].shape[-2] - s1.shape[-2], 0))
        S = torch.cat([s1, sx[1]], dim=1)[:, :, :32, :]
        out_dim = S.shape[1:3]
        return out_dim
        

@gin.configurable
class MedleySolosDB(Dataset):
    def __init__(self, 
                 data_dir='/import/c4dm-datasets/medley-solos-db/', 
                 subset='training',
                 feature='jtfs'):
        super().__init__()
        
        self.msdb = msdb.Dataset(data_dir)
        self.audio_dir = os.path.join(data_dir, 'audio')
        self.csv_dir = os.path.join(data_dir, 'annotation')
        self.subset = subset

        self.feature = feature.split('_')[0]
        self.feature_spec = feature.split('_')[1:]

        # if 'jtfs' in feature or 'scat1d' in feature:
        #     feature_dir = os.path.join(data_dir, feature)
        #     self.feature_dir = os.path.join(feature_dir, subset)
        # elif feature == 'cqt':
        #     self.feature_dir = None
        feature_dir = os.path.join(data_dir, feature)
        self.feature_dir = os.path.join(feature_dir, subset)
        
        df = pd.read_csv(os.path.join(self.csv_dir, 'Medley-solos-DB_metadata.csv'))
        self.df = df.loc[df['subset'] == subset]
        self.df.reset_index(inplace = True)
        
    def build_fname(self, df_item, ext='.npy'):
        uuid = df_item['uuid4']
        instr_id = df_item['instrument_id']
        subset = df_item['subset']
        if self.feature == 'jtfs' and '3D' in self.feature_spec:
            s1 = f'Medley-solos-DB_{subset}-{instr_id}_{uuid}_S1{ext}'
            s2 = f'Medley-solos-DB_{subset}-{instr_id}_{uuid}_S2{ext}'
            return s1, s2
        else:
            return f'Medley-solos-DB_{subset}-{instr_id}_{uuid}{ext}'

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        y = int(item['instrument_id'])

        if self.feature == 'jtfs' and '3D' in self.feature_spec:
            s1_fname, s2_fname = self.build_fname(item, '.npy') 
            s1 = np.load(os.path.join(self.feature_dir, s1_fname))
            s2 = np.load(os.path.join(self.feature_dir, s2_fname))
            Sx = (s1, s2)
            return Sx, y
        elif self.feature == 'scat1d' or (self.feature == 'jtfs' and '2D' in self.feature_spec) or self.feature == 'cqt':
            fname = self.build_fname(item, '.npy') 
            Sx = np.load(os.path.join(self.feature_dir, fname))
            return Sx, y
        else:
            fname = self.build_fname(item, '.wav') 
            audio, _ = msdb.load_audio(os.path.join(self.audio_dir, fname))
            x = audio
            return x, y, fname

    def __len__(self):
        return len(self.df)


@gin.configurable
class MedleyDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str = '/import/c4dm-datasets/medley-solos-db/', 
                 batch_size: int = 32,
                 feature='jtfs'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.feature = feature

    def setup(self, stage: Optional[str] = None):
        self.train_ds = MedleySolosDB(self.data_dir, subset='training', feature=self.feature)
        self.val_ds = MedleySolosDB(self.data_dir, subset='validation', feature=self.feature)
        self.test_ds = MedleySolosDB(self.data_dir, subset='test', feature=self.feature)

    def train_dataloader(self):
        return DataLoader(self.train_ds, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          drop_last=True,
                          num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_ds, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          drop_last=True,
                          num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_ds, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          drop_last=True, 
                          num_workers=1)


class LeNet(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=5, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=256, kernel_size=5, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=84),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x):
        return self.classifier(self.feature_extractor(x).flatten(1))


class LeNet1D(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.feature_extractor = nn.Sequential(            
            nn.Conv1d(in_channels=in_channels, out_channels=512, kernel_size=5, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=84),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x):
        return self.classifier(self.feature_extractor(x).flatten(1))

from torchvision.ops import StochasticDepth

class SqueezeExciteNd(nn.Module):
    def __init__(self, num_channels, r=16):
        """num_channels: No of input channels
           r: By how much should the num_channels should be reduced
        """
        super().__init__()
        num_channels_reduced = num_channels // r
        assert r <= num_channels, (r, num_channels)

        self.r = r
        # nn.AdaptiveAvgPool2d
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, *spatial = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1
                                           ).mean(dim=-1)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(
            squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        view_shape = (batch_size, num_channels) + (1,) * len(spatial)
        output_tensor = torch.mul(input_tensor, fc_out_2.view(*view_shape))

        return output_tensor

class ConvNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, act=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, 
                   out_channels, 
                   kernel_size=kernel_size, 
                   stride=stride, 
                   padding=padding, 
                   groups=groups,
                   bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU() if act else nn.Identity()
        )

    def forward(self, input_tensor):
        return self.block(input_tensor)



class MBConvN(nn.Module):
  """MBConv with an expansion factor of N, plus squeeze-and-excitation"""
  def __init__(self, n_in, n_out, expansion_factor,
               kernel_size=3, stride=1, r=24, p=0):
    super().__init__()

    padding = (kernel_size - 1) // 2
    expanded = expansion_factor * n_in
    self.skip_connection = (n_in == n_out) and (stride == 1)

    self.expand_pw = nn.Identity() if (expansion_factor == 1) else ConvNormActivation(n_in, expanded, kernel_size=1)
    self.depthwise = ConvNormActivation(expanded, expanded, kernel_size=kernel_size, 
                                        stride=stride, padding=padding, groups=expanded)
    self.se = SqueezeExciteNd(expanded, r=r)
    self.reduce_pw = ConvNormActivation(expanded, n_out, kernel_size=1, act=False)
    self.dropsample = StochasticDepth(p, mode='row')
  
  def forward(self, x):
    residual = x

    x = self.expand_pw(x)
    x = self.depthwise(x)
    x = self.se(x)
    x = self.reduce_pw(x)

    if self.skip_connection:
      x = self.dropsample(x)
      x = x + residual

    return x

class MBConv1(MBConvN):
  def __init__(self, n_in, n_out, kernel_size=3, stride=1, r=24, p=0):
    super().__init__(n_in, n_out, expansion_factor=1,
                     kernel_size=kernel_size, stride=stride,
                     r=r, p=p)
    
 
class MBConv6(MBConvN):
  def __init__(self, n_in, n_out, kernel_size=3, stride=1, r=24, p=0):
    super().__init__(n_in, n_out, expansion_factor=6,
                     kernel_size=kernel_size, stride=stride,
                     r=r, p=p)


class EfficientNet1d(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvNormActivation(in_channels, 32, kernel_size=3, stride=2), 
            MBConv1(32, 16, kernel_size=3, r=4, p=0.0),
            MBConv6(16, 24, kernel_size=3, stride=2, r=24, p=0.0125),
            MBConv6(24, 24, kernel_size=3, stride=1, r=24, p=0.025),
            MBConv6(24, 40, kernel_size=5, stride=2, r=24, p=0.0375),
            MBConv6(40, 240, kernel_size=5, stride=1, r=24, p=0.05), 
            MBConv6(240, 80, kernel_size=3, stride=2, r=24, p=0.0625), 
            MBConv6(80, 80, kernel_size=3, stride=1, r=24, p=0.075), 
            MBConv6(80, 80, kernel_size=3, stride=1, r=24, p=0.0875), 
            MBConv6(80, 112, kernel_size=5, stride=1, r=24, p=0.1), 
            MBConv6(112, 112, kernel_size=5, stride=1, r=24, p=0.1125), 
            MBConv6(112, 112, kernel_size=5, stride=1, r=24, p=0.125), 
            MBConv6(112, 192, kernel_size=5, stride=2, r=24, p=0.1375), 
            MBConv6(192, 192, kernel_size=5, stride=1, r=24, p=0.15), 
            MBConv6(192, 192, kernel_size=5, stride=1, r=24, p=0.1625), 
            MBConv6(192, 192, kernel_size=5, stride=1, r=24, p=0.175), 
            MBConv6(192, 320, kernel_size=3, stride=1, r=24, p=0.1875), 
            ConvNormActivation(320, 1280, kernel_size=1, stride=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes, bias=True)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x).squeeze(-1)
        y = self.classifier(x)
        return y