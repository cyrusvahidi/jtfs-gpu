# run configs ################################################################
# random seed
SEED = 0
# `None`: train -- str: load model to test
TEST_PATH = None
# number of train epochs
N_EPOCHS = 20
# batch size
BATCH_SIZE = 32
# wandb
LOG = False
# number of samples per epoch
EPOCH_SIZE = 8192
# path to config file
GIN_CONFIG_FILE = 'scripts/gin/config.gin'

# code #######################################################################
import gin
import pytorch_lightning as pl, fire
pl.seed_everything(SEED)
from pytorch_lightning.callbacks import (
    RichProgressBar, ModelCheckpoint, TQDMProgressBar)
from pytorch_lightning.loggers import WandbLogger

from kymjtfs.cnn import MedleySolosClassifier, MedleyDataModule
from kymjtfs.utils import make_abspath
from pprint import pprint

# import warnings
# warnings.filterwarnings("ignore")


def run_model():
    gin.parse_config_file(make_abspath(GIN_CONFIG_FILE))

    # build model & dataset ##################################################
    progbar_callback = TQDMProgressBar(refresh_rate=50)
    wandb_logger = WandbLogger(project='cqt') if LOG else None
    checkpoint_kw = dict(
        filename='cqt-{step}-val_acc{val/acc:.3f}-val_loss{val/loss:.3f}',
        monitor='val/acc',
        mode='max',
        every_n_epochs=1,
        save_top_k=-1,
    )
    checkpoint_cb = ModelCheckpoint(**checkpoint_kw)
    n_batches_train = EPOCH_SIZE // BATCH_SIZE

    trainer = pl.Trainer(gpus=-1,
                         max_epochs=N_EPOCHS,
                         callbacks=[progbar_callback, checkpoint_cb],
                         # fast_dev_run=True,
                         limit_train_batches=n_batches_train,
                         logger=wandb_logger)
    dataset = MedleyDataModule(batch_size=BATCH_SIZE)

    # train / load to test
    if TEST_PATH is None:
        model = MedleySolosClassifier(n_batches_train=n_batches_train)
        trainer.fit(model, dataset)
    else:
        model = MedleySolosClassifier.load_from_checkpoint(TEST_PATH)

    # test ###################################################################
    x = trainer.test(model, dataset, verbose=False)
    results = {'acc_macro': x[0]['acc_macro'],
               'acc_classwise': [float(i) for i in x[0]['acc_classwise'].values()],
               'val_acc': x[0].get('val_acc', -1),
               'val_loss': x[0].get('val_loss', -1)}
    pprint(results)
    if TEST_PATH is not None:
        print(TEST_PATH)


def main():
  fire.Fire(run_model)


if __name__ == "__main__":
    main()
