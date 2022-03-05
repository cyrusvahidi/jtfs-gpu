from kymjtfs.cnn import MedleySolosClassifier, MedleyDataModule

import pytorch_lightning as pl, fire

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from joblib import Memory

def run_train(n_epochs = 200, 
              batch_size = 4, 
              cachedir='/import/c4dm-04/cv'):
    memory = Memory(cachedir, verbose=0)
  
    early_stop_callback = EarlyStopping(monitor="val/loss", 
                                        min_delta=0.00, 
                                        patience=5, 
                                        verbose=False, 
                                        mode="max")
    wandb_logger = WandbLogger('jtfs_dafx')
    trainer = pl.Trainer(gpus=-1, 
                        max_epochs=n_epochs,
                        progress_bar_refresh_rate=1, 
                        checkpoint_callback=True,
                        callbacks=[early_stop_callback],
                        fast_dev_run=True,
                        overfit_batches=5,
                        logger=wandb_logger)
    model = MedleySolosClassifier()
    dataset = MedleyDataModule(jtfs=model.jtfs, batch_size=batch_size) 
    trainer.fit(model, dataset)
    trainer.test(model, dataset)


def main():
  fire.Fire(run_train)


if __name__ == "__main__":
    main()