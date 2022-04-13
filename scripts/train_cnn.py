import gin, os
import pytorch_lightning as pl, fire
pl.seed_everything(1)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    RichProgressBar, ModelCheckpoint, TQDMProgressBar)
import torch, torch.nn as nn
from pytorch_lightning.loggers import WandbLogger

from kymjtfs.cnn import MedleySolosClassifier, MedleyDataModule
from kymjtfs.utils import make_abspath
from pprint import pprint

# import warnings
# warnings.filterwarnings("ignore")

n_epochs = 30
batch_size = 32
log = False
epoch_size = 8192
n_batches_train = epoch_size // batch_size
gin_config_file = 'gin/config.gin'
gin.parse_config_file(make_abspath(gin_config_file))

progbar_callback = TQDMProgressBar(refresh_rate=50)
wandb_logger = WandbLogger(project='kymatio-jtfs') if log else None

checkpoint_kw = dict(
    filename='jtfs_3D_J13-{step}-val_acc{val/acc:.3f}',
    monitor='val/acc',
    mode='max',
    every_n_epochs=1,
    save_top_k=-1,
)
checkpoint_cb = ModelCheckpoint(**checkpoint_kw)

path = None


path = r"C:\Desktop\School\Deep Learning\DL_Code\kymatio-jtfs\scripts\checkpoints\jtfs_3D_J13-step=2366-val_accval\acc=0.838.ckpt"
trainer = pl.Trainer(gpus=-1,
                     max_epochs=n_epochs,
                     callbacks=[progbar_callback, checkpoint_cb],
                     # fast_dev_run=True,
                     limit_train_batches=n_batches_train,
                     logger=wandb_logger)
dataset = MedleyDataModule(batch_size=batch_size)

if path is None:
    model = MedleySolosClassifier(n_batches_train=n_batches_train)
else:
    model = MedleySolosClassifier.load_from_checkpoint(path)

if path is None:
    trainer.fit(model, dataset)

x = trainer.test(model, dataset, verbose=False)
results = {'acc_macro': x[0]['acc_macro'],
           'acc_classwise': [float(i) for i in x[0]['acc_classwise'].values()],
           'val_acc': x[0].get('val_acc', -1),
           'val_loss': x[0].get('val_loss', -1)}
pprint(results)
print(path)
