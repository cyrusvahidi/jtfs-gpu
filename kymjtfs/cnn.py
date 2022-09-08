from typing import Optional, Union

import os, pandas as pd, numpy as np, torch, librosa, gin
import pytorch_lightning as pl
import mirdata.datasets.medley_solos_db as msdb
import torchvision.models as models

from torch.nn import functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import AmplitudeToDB
from torchvision.ops import StochasticDepth
from torchmetrics import Accuracy, ClasswiseWrapper
from pytorch_lightning.core.lightning import LightningModule
from nnAudio.features import CQT
from kymatio.torch import TimeFrequencyScattering1D
from kymjtfs.utils import make_abspath


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
                 stats_dir='scripts/import/c4dm-datasets/medley-solos-db/',
                 csv=('scripts/import/c4dm-datasets/medley-solos-db/annotation/'
                      'Medley-solos-DB_metadata.csv'),
                 feature='jtfs',
                 learn_adalog = True,
                 std = False,
                 n_batches_train = None):
        super().__init__()

        self.jtfs_kwargs = jtfs_kwargs
        self.lr = lr
        self.feature = feature.split('_')[0]
        self.feature_spec = feature.split('_')[1:]
        self.learn_adalog = learn_adalog
        self.std = std
        self.n_batches_train = n_batches_train

        df = pd.read_csv(make_abspath(csv))
        classes = df[['instrument', 'instrument_id']
                     ].value_counts().index.to_list()
        classes = [x[0] for x in sorted(classes, key=lambda x: x[1])]

        self.acc_metric = Accuracy(num_classes=len(classes), average=average)
        self.acc_metric_macro = Accuracy(num_classes=len(classes), average='macro')

        self.classwise_acc = ClasswiseWrapper(
            Accuracy(num_classes=len(classes), average=None), labels=classes)
        self.loss = nn.CrossEntropyLoss(weight=self.get_class_weight(csv))

        self.val_acc = None
        self.val_loss = None

        self.is_2d_conv = (self.feature == 'cqt' or self.feature == 'jtfs')

        stats_dir = make_abspath(stats_dir)
        if self.feature in ('jtfs', 'scat1d'):
            stats_dir = os.path.join(stats_dir, feature)

            def load_as_tensor(path):
                return torch.tensor(np.load(os.path.join(stats_dir, path)))

            if self.feature == 'jtfs':
                if not self.std:
                    self.mu = load_as_tensor('stats/mu.npy')
                # renormalization
                self.mu_s1 = load_as_tensor('stats/mu_s1.npy')
                self.mu_s2 = load_as_tensor('stats/mu_s2.npy')

                # standardization
                self.mu_z_s1 = load_as_tensor('stats/mu_z_s1.npy')
                self.mu_z_s2 = load_as_tensor('stats/mu_z_s2.npy')
                self.std_z_s1 = load_as_tensor('stats/std_z_s1.npy')
                self.std_z_s2 = load_as_tensor('stats/std_z_s2.npy')
            else:
                self.mu = load_as_tensor('stats/mu.npy')
                if self.std:
                    self.mu_z = load_as_tensor('stats/mu_z.npy')
                    self.std_z = load_as_tensor('stats/std_z.npy')

            if self.feature == 'jtfs':
                s1_channels = 4
                self.s1_conv1 = nn.Sequential(
                    nn.Conv2d(1, s1_channels, kernel_size=(16, 1)),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size=(4, 1), padding=(2, 0))
                )
                if self.std:
                    self.n_channels = len(self.mu_s2) + s1_channels
                else:
                    self.n_channels = len(self.mu) + (s1_channels - 1)
            elif (self.feature == 'scat1d' or
                  (self.feature == 'jtfs' and '2D' in self.feature_spec)):
                self.n_channels = len(self.mu)

            self.c = c
            if self.learn_adalog:
                if self.std and self.feature == 'jtfs':
                    self.register_parameter('eps',
                                            nn.Parameter(torch.randn(
                                                len(self.mu_s1) + len(self.mu_s2))))
                else:
                    self.register_parameter('eps',
                                            nn.Parameter(torch.randn(len(self.mu))))
        elif feature == 'cqt':
            self.n_channels = 1
            stats_dir = os.path.join(stats_dir, feature)
            self.mu = torch.tensor(np.load(os.path.join(stats_dir,
                                                        'stats/mu.npy')))
            self.std = torch.tensor(np.load(os.path.join(stats_dir,
                                                         'stats/std.npy')))
            # self.cqt = CQT(sr=44100, n_bins=96, hop_length=256, fmin=32.7)
            # self.a_to_db = AmplitudeToDB(stype = 'magnitude')

        self.bn = (nn.BatchNorm2d(self.n_channels) if self.is_2d_conv else
                   nn.BatchNorm1d(self.n_channels))

        self.setup_cnn(len(classes))

        self.automatic_optimization = False

    def setup_cnn(self, num_classes):
        if self.is_2d_conv:
            if self.feature == 'jtfs':
                self.conv_net = stuff2D(self.n_channels, num_classes=num_classes,
                                        se_r=16, c_mult_init=1)
            elif self.feature == 'cqt':
                self.conv_net = stuff2D(self.n_channels, num_classes=num_classes,
                                        se_r=4, c_mult_init=64)
        else:
            self.conv_net = stuff1D(self.n_channels, num_classes=num_classes)

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
                if self.std:
                    c1, c2 = (c[None, :len(self.mu_s1), None],
                              c[None, len(self.mu_s1):, None, None])
                else:
                    c1, c2 = c[None, :1, None], c[None, 1:, None, None]

            if self.std:
                s1 = s1[:, 1:, :] / (c1 * self.mu_s1[None, :, None].type_as(s1)
                                     + 1e-8)
                s1 = torch.log1p(s1)
                s1 = (s1 - self.mu_z_s1[None, :, None].type_as(s1)
                      ) / self.std_z_s1[None, :, None].type_as(s1)

                s2 = s2 / (c2 * self.mu_s2[None, :, None, None].type_as(s2)
                           + 1e-8)
                s2 = torch.log1p(s2)
                s2 = (s2 - self.mu_z_s2[None, :, None, None].type_as(s2)
                      ) / self.std_z_s2[None, :, None, None].type_as(s2)
            else:
                s1 = s1 / (c1 * self.mu[None, :, None].type_as(s1) + 1e-8)
                s2 = s2 / (c2 * self.mu_s2[None, :, None, None].type_as(s2)
                           + 1e-8)
                s2 = torch.log1p(s2)

            # s1 learnable frequential filter
            s1_conv = self.s1_conv1(s1.unsqueeze(1))
            s1_conv = F.pad(s1_conv, (0, 0, s2.shape[-2] - s1_conv.shape[-2], 0))
            sx = torch.cat([s1_conv, s2], dim=1)
            # log1p and batch norm
            if not self.std:
                sx = torch.log1p(sx)
            sx = self.bn(sx)

        elif (self.feature == 'scat1d' or
              (self.feature == 'jtfs' and not self.is_2d_conv)):
            Sx = x
            c = self.get_c().type_as(Sx)
            sx = torch.log1p(Sx / (c[None, :, None] *
                                   self.mu[None, :, None].type_as(Sx) + 1e-8))
            if self.std:
                sx = (sx - self.mu_z[None, :, None].type_as(sx)
                      ) / self.std_z[None, :, None].type_as(sx)
            sx = self.bn(sx)

        elif self.feature == 'cqt':
            # X = self.a_to_db(self.cqt(x))
            mu = self.mu[None, :, None].type_as(x)
            std = self.std[None, :, None].type_as(x)
            x = (x - mu) / std
            sx = F.avg_pool2d(x, kernel_size=(3, 8))
            sx = sx.unsqueeze(1)
            sx = self.bn(sx)

        # conv net
        y = self.conv_net(sx)

        return y

    def step(self, batch, fold):
        Sx, y = batch
        logits = self(Sx)

        loss = self.loss(logits, y)

        return {'loss': loss,
                'logits': logits,
                'y': y}

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        def closure():
            opt.zero_grad()
            self.manual_backward(loss)
            return loss

        self.update_lr(batch_idx)
        opt.step(closure=closure)

        return {'loss': loss,
                'logits': logits,
                'y': y}

    def validation_step(self, batch, batch_idx):
        return self.step(batch, fold='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, fold='test')

    def training_epoch_end(self, outputs):
        logits = torch.cat([x['logits'] for x in outputs]).softmax(dim=-1)
        y = torch.cat([x['y'] for x in outputs])
        acc_macro = self.acc_metric_macro(logits, y)
        loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('train/acc', acc_macro)
        self.log('train/loss', loss, prog_bar=True)

        self.reset_metrics()

    def validation_epoch_end(self, outputs):
        logits = torch.cat([x['logits'] for x in outputs]).softmax(dim=-1)
        y = torch.cat([x['y'] for x in outputs])
        acc_macro = self.acc_metric_macro(logits, y)

        self.val_acc = acc_macro
        self.val_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('val/acc', self.val_acc, on_step=False,
                 prog_bar=True, on_epoch=True)
        self.log('val/loss', self.val_loss, on_step=False,
                 prog_bar=True, on_epoch=True)

        self.reset_metrics()

    def test_epoch_end(self, outputs):
        logits = torch.cat([x['logits'] for x in outputs]).softmax(dim=-1)
        y = torch.cat([x['y'] for x in outputs])
        acc_macro = self.acc_metric_macro(logits, y)
        # acc_classwise = self.classwise_acc(logits, y)

        bin_counts = torch.bincount(y)
        classwise_acc = [torch.zeros(n) for n in bin_counts]
        class_counts = [0 for _ in bin_counts]
        preds = logits.argmax(dim=-1)
        for i, p in enumerate(preds):
            score = float(preds[i] == y[i])
            classwise_acc[y[i]][class_counts[y[i]]] = score
            class_counts[y[i]] += 1

        acc_classwise = {i: float(acc.mean())
                         for i, acc in enumerate(classwise_acc)}

        if self.val_acc is not None:
            self.log(f'val_acc', self.val_acc)
            self.log(f'val_loss', self.val_loss)
        self.log('acc_macro', acc_macro)
        self.log('acc_classwise', acc_classwise)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                weight_decay=1e-1)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=1, eta_min=1e-8,
            last_epoch=-1, verbose=0)
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                },
            }
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
        #                                                        factor=0.5,
        #                                                        patience=5,
        #                                                        mode='min')
        # return {'optimizer': opt, 'lr_scheduler': scheduler,
        #         'monitor':  'val/loss'}

    def update_lr(self, batch_idx):
        sch = self.lr_schedulers()

        warmup_epochs = 3
        warmup_len = self.n_batches_train * warmup_epochs
        total_step = self.trainer.current_epoch * self.n_batches_train + batch_idx
        if total_step >= warmup_len:
            epoch_frac = total_step / self.n_batches_train
        else:
            # LR warmup for first epoch
            # `batch_idx + 1` to not start with `1` when `batch_idx == 0`
            epoch_frac = 1 - (total_step + 1) / warmup_len
        sch.step(epoch_frac)
        return sch

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=3,
            gradient_clip_algorithm='norm',
        )

    def reset_metrics(self):
        self.acc_metric_macro.reset()
        self.classwise_acc.reset()

    def get_c(self):
        c = (self.c * torch.exp(torch.tanh(self.eps)) if self.learn_adalog else
             torch.tensor([self.c]))
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
                 data_dir='import/c4dm-datasets/medley-solos-db/',
                 subset='training',
                 feature='jtfs',
                 out_dir_to_skip=None,  # resuming mechanism
                 ):
        super().__init__()

        data_dir = make_abspath(data_dir)

        self.msdb = msdb.Dataset(data_dir)
        self.audio_dir = os.path.join(data_dir, 'audio')
        self.csv_dir = os.path.join(data_dir, 'annotation')
        self.subset = subset
        if out_dir_to_skip is None:
            self.out_dir_to_skip = None
        else:
            self.out_dir_to_skip = os.path.join(out_dir_to_skip, subset)

        self.feature = feature.split('_')[0]
        self.feature_spec = feature.split('_')[1:]

        # if 'jtfs' in feature or 'scat1d' in feature:
        #     feature_dir = os.path.join(data_dir, feature)
        #     self.feature_dir = os.path.join(feature_dir, subset)
        # elif feature == 'cqt':
        #     self.feature_dir = None
        feature_dir = os.path.join(data_dir, feature)
        self.feature_dir = os.path.join(feature_dir, subset)

        df = pd.read_csv(os.path.join(self.csv_dir,
                                      'Medley-solos-DB_metadata.csv'))
        self.df = df.loc[df['subset'] == subset]
        self.df.reset_index(inplace = True)

        if (self.out_dir_to_skip is not None
                and os.path.isdir(self.out_dir_to_skip)):
            self.out_names_done = os.listdir(self.out_dir_to_skip)
        else:
            self.out_names_done = None

    def build_fname(self, df_item, ext='.npy'):
        uuid = df_item['uuid4']
        instr_id = df_item['instrument_id']
        subset = df_item['subset']
        if self.feature == 'jtfs':
            s1 = f'Medley-solos-DB_{subset}-{instr_id}_{uuid}_S1{ext}'
            s2 = f'Medley-solos-DB_{subset}-{instr_id}_{uuid}_S2{ext}'
            return s1, s2
        else:
            return f'Medley-solos-DB_{subset}-{instr_id}_{uuid}{ext}'

    def build_fname2(self, df_item, ext='.npy'):
        uuid = df_item['uuid4']
        instr_id = df_item['instrument_id']
        subset = df_item['subset']
        s1 = f'Medley-solos-DB_{subset}-{instr_id}_{uuid}_S1{ext}'
        s2 = f'Medley-solos-DB_{subset}-{instr_id}_{uuid}_S2{ext}'
        return s1, s2

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        y = int(item['instrument_id'])

        if self.feature == 'jtfs':
            s1_fname, s2_fname = self.build_fname(item, '.npy')

            s1 = np.load(os.path.join(self.feature_dir, s1_fname))
            s2 = np.load(os.path.join(self.feature_dir, s2_fname))
            Sx = (s1, s2)
            return Sx, y
        elif (self.feature in ('scat1d', 'cqt')):
            fname = self.build_fname(item, '.npy')
            Sx = np.load(os.path.join(self.feature_dir, fname))
            return Sx, y
        else:
            fname = self.build_fname(item, '.wav')
            s1_fname, s2_fname = self.build_fname2(item, '.npy')
            if (self.out_names_done is not None and
                    os.path.basename(s2_fname) in self.out_names_done):
                print(end='.')
                return

            audio, _ = msdb.load_audio(os.path.join(self.audio_dir, fname))
            x = audio
            return x, y, fname

    def __len__(self):
        return len(self.df)


@gin.configurable
class MedleyDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = 'import/c4dm-datasets/medley-solos-db/',
                 batch_size: int = 32,
                 feature='jtfs',
                 out_dir_to_skip=None):
        super().__init__()
        data_dir = make_abspath(data_dir)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.feature = feature
        self.out_dir_to_skip = out_dir_to_skip

    def setup(self, stage: Optional[str] = None):
        self.train_ds = MedleySolosDB(self.data_dir, subset='training',
                                      feature=self.feature,
                                      out_dir_to_skip=self.out_dir_to_skip)
        self.val_ds = MedleySolosDB(self.data_dir, subset='validation',
                                    feature=self.feature,
                                    out_dir_to_skip=self.out_dir_to_skip)
        self.test_ds = MedleySolosDB(self.data_dir, subset='test',
                                     feature=self.feature,
                                     out_dir_to_skip=self.out_dir_to_skip)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=True,
                          num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=True,
                          num_workers=0)


class LeNet(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=5,
                      stride=1),
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


class stuff2D(nn.Module):
    def __init__(self, in_channels, num_classes, dense_dim=64, drop_rate=.5,
                 se_r=16, c_mult_init=1):
        super().__init__()
        """
        JTFS:
            se_r=16; c_ref = in_channels * 2
            MedleySolosClassifier.std = 1
        CQT:
            se_r=4;  c_ref = in_channels * 128
            MedleySolosClassifier.std = 1
        """
        ckw = dict(stride=(1, 1), bias=False, padding='same')
        c_ref = in_channels * c_mult_init
        C0, C1, C2 = c_ref, 2*c_ref, 4*c_ref

        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=C0,
                               kernel_size=(7, 7), **ckw)
        self.bn0 = nn.BatchNorm2d(C0)
        self.se0 = None
        self.mp0 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv1 = nn.Conv2d(in_channels=C0, out_channels=C1,
                               kernel_size=(5, 5), **ckw)
        self.bn1 = nn.BatchNorm2d(C1)
        self.se1 = SqueezeExciteNd(num_channels=C1, r=se_r)
        self.mp1 = None
        # self.mp1 = nn.MaxPool2d(kernel_size=(2, 1))

        self.conv2 = nn.Conv2d(in_channels=C1, out_channels=C2,
                               kernel_size=(3, 3), **ckw)
        self.bn2 = nn.BatchNorm2d(C2)
        self.se2 = SqueezeExciteNd(num_channels=C2, r=se_r)

        self.relu = nn.ReLU()
        self.gap = GlobalAveragePooling(flatten=True)

        self.fc = nn.Linear(in_features=C2, out_features=dense_dim)
        self.dp = nn.Dropout(drop_rate)
        self.fc_out = nn.Linear(in_features=dense_dim, out_features=num_classes)

    def forward_features(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        if self.se0 is not None:
            x = self.se0(x)
        x = self.relu(x)
        x = self.mp0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.se1(x)
        x = self.relu(x)
        if self.mp1 is not None:
            x = self.mp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se2(x)
        x = self.relu(x)

        x = self.gap(x)
        return x

    def classifier(self, x):
        x = self.dp(x)

        x = self.fc(x)
        x = self.relu(x)

        x = self.fc_out(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


class stuff1D(nn.Module):
    def __init__(self, in_channels, num_classes, dense_dim=64, drop_rate=.5):
        super().__init__()
        c_ref = in_channels // 4
        ckw = dict(stride=1, bias=False, padding='same')
        C0, C1, C2 = c_ref//2, c_ref, 2*c_ref

        self.conv0 = nn.Conv1d(in_channels=in_channels, out_channels=C0,
                               kernel_size=7, **ckw)
        self.bn0 = nn.BatchNorm1d(C0)
        self.se0 = None
        self.mp0 = nn.MaxPool1d(kernel_size=2)

        self.conv1 = nn.Conv1d(in_channels=C0, out_channels=C1,
                               kernel_size=5, **ckw)
        self.bn1 = nn.BatchNorm1d(C1)
        self.se1 = SqueezeExciteNd(num_channels=C1)
        self.mp1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=C1, out_channels=C2,
                               kernel_size=3, **ckw)
        self.bn2 = nn.BatchNorm1d(C2)
        self.se2 = SqueezeExciteNd(num_channels=C2)

        self.relu = nn.ReLU()
        self.gap = GlobalAveragePooling(flatten=True)

        self.fc = nn.Linear(in_features=C2, out_features=dense_dim)
        self.dp = nn.Dropout(drop_rate)
        self.fc_out = nn.Linear(in_features=dense_dim, out_features=num_classes)

    def forward_features(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        if self.se0 is not None:
            x = self.se0(x)
        x = self.relu(x)
        x = self.mp0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.se1(x)
        x = self.relu(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se2(x)
        x = self.relu(x)

        x = self.gap(x)
        return x

    def classifier(self, x):
        x = self.dp(x)

        x = self.fc(x)
        x = self.relu(x)

        x = self.fc_out(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


class LeNet1D(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=512, kernel_size=5,
                      stride=1),
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


class GlobalAveragePooling(nn.Module):
    def __init__(self, flatten=True):
        super(GlobalAveragePooling, self).__init__()
        self.flatten = flatten
        self.keepdim = bool(not self.flatten)

    def forward(self, x):
        return x.mean(dim=tuple(range(2, x.ndim)), keepdim=self.keepdim)


class ConvNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, groups=1, act=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels,
                   out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
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

    self.expand_pw = (nn.Identity() if (expansion_factor == 1) else
                      ConvNormActivation(n_in, expanded, kernel_size=1))
    self.depthwise = ConvNormActivation(expanded, expanded,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=padding, groups=expanded)
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
