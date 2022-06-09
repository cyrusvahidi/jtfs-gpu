import torch

from .utils import generate_am_chirp


@gin.configurable
class AMChirpSynth(LightningModule):
    def __init__(self, 
                 c = 1e-1,
                 jtfs_kwargs = {'shape': (2**16, ),
                                'J': 12,
                                'Q': 16,
                                'F': 4,
                                'T': 2**11},
                 lr=1e-3):
        super().__init__()

        self.jtfs_kwargs = jtfs_kwargs
        self.lr = lr 
        self.c = torch.tensor([c])

        self.setup_cnn()                                                 
         
    def setup_cnn(self):
        pass

    def setup_jtfs(self):
        self.jtfs = TimeFrequencyScattering1D(
            **self.jtfs_kwargs,
            average_fr=False,
            max_pad_factor=3, 
            max_pad_factor_fr=3)
        
        n_channels = self._get_jtfs_out_dim()
        
        self.jtfs_dim = self._get_jtfs_out_dim()
        self.jtfs_channels = self.jtfs_dim[0]

    def forward(self, x):
        pass
 
    def step(self, batch, fold):
        Sx, y = batch
        logits = self(Sx)

        loss, acc = self.loss(logits, y), self.acc_metric(logits, y)
        class_acc = self.classwise_acc(logits, y)
        class_acc = {k: float(v.detach()) for k, v in class_acc.items()}
        
        self.log(f'{fold}/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{fold}/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        if fold == 'test':
            macro_avg = self.acc_metric_macro(logits, y)
            self.log(f'{fold}/avg_macro', macro_avg, on_step=False, on_epoch=True, prog_bar=True)
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
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, 
                                                      base_lr=self.lr, 
                                                      max_lr=self.lr * 4, 
                                                      step_size_up=1024, 
                                                      mode='triangular', 
                                                      gamma=1.0, 
                                                      cycle_momentum=False)
        return {'optimizer': opt, 'lr_scheduler': scheduler}
    
    def _get_jtfs_out_dim(self):
        dummy_in = torch.randn(self.in_shape).cuda()
        sx = self.jtfs(dummy_in)
        out_dim = S.shape[-2]
        return out_dim
        

@gin.configurable
class AMChirpDataset(Dataset):
    def __init__(self, 
                 subset='training'):
        super().__init__()

        self.subset = subset

        # self.n_samples = 
        
    def generate()

    def __getitem__(self, idx):

    def __len__(self):
        return self.n_samples


@gin.configurable
class MedleyDataModule(pl.LightningDataModule):
    def __init__(self, 
                 batch_size: int = 32,):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.train_ds = AMChirpDataset(subset='training')
        self.val_ds = AMChirpDataset(subset='validation')
        self.test_ds = AMChirpDataset(subset='test')

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
