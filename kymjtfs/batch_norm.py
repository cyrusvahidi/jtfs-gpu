import torch 
import torch.nn as nn


class ScatteringBatchNorm(nn.Module):
    ''' A general purpose batch normalization layer,
        designed particularly for scattering paths.
    
    Args:
        path_shape: the shape of scattering paths and lambda (alpha x beta, lambda)
        c: an optionally learnable multiplicative featurewise constant, used
            before taking the logarithm to gaussianize the histogram of scattering 
            path values.
    '''
    def __init__(self, path_shape, c: float = None, eval_mode: bool = False):
        super().__init__()

        self.path_shape = path_shape

        self.register_buffer("mu", torch.zeros(self.path_shape))
        # self.c = torch.nn.Parameter(torch.ones(num_features, ) * 5.0) if not c else c

    def _check_input_dim(self, x):
        if x.shape[1:3] != self.path_shape:
            raise ValueError(
                "expected second dim to equal feature dim: {} (got {} input)".format(self.path_shape, x.shape[1:3])
            )

    def reset_stats(self):
        pass

    def forward(self, sx):
        self._check_input_dim(sx)
        
        if self.training:
            batch_size = sx.shape[0]
            batch_mu =  sx.mean(dim=0).mean(dim=-1)
            
            self.mu = (1 / (batch_size + 1)) * self.mu.detach() + (batch_size / (batch_size + 1)) * batch_mu

            sx = sx / (1e-3 + self.mu.view(1, self.mu.shape[0], -1, 1))
        return sx 