from kymjtfs.cnn import MedleySolosClassifier, MedleyDataModule

import pytorch_lightning as pl, fire

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import warnings

warnings.filterwarnings("ignore")

def run_train(n_epochs = 20, 
              batch_size = 32):
    early_stop_callback = EarlyStopping(monitor="val/loss_epoch", 
                                        min_delta=0.00, 
                                        patience=5, 
                                        verbose=True, 
                                        mode="min")
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(gpus=-1, 
                        max_epochs=n_epochs,
                        progress_bar_refresh_rate=1, 
                        enable_checkpointing=True,
                        # callbacks=[early_stop_callback],
                        # fast_dev_run=True,
                        logger=wandb_logger)
    model = MedleySolosClassifier()
    dataset = MedleyDataModule(batch_size=batch_size) 
    trainer.fit(model, dataset)
    trainer.test(model, dataset)


def main():
  fire.Fire(run_train)


if __name__ == "__main__":
    main()